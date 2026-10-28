"""Offline checks for thread-safe Board writes, the Modulator loop, and warm starts."""

import serial
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

import my_patterns
from my_patterns import Board, Modulator, burst, hold, phases_trap, pattern_for_kind


class FakeSerial:
    """In-memory pyserial stand-in. 分块写入并穿插 sleep,无锁时并发写会交错。"""

    def __init__(self, port=None, baudrate=9600, timeout=None, write_timeout=None, **kw):
        self.port, self.baudrate = port, baudrate
        self.stream = bytearray()
        self._lock = threading.Lock()
        self.write_delay = 0.0

    def write(self, data):
        data = bytes(data)
        for i in range(0, len(data), 64):
            if self.write_delay:
                time.sleep(self.write_delay)
            with self._lock:
                self.stream.extend(data[i:i + 64])
        return len(data)

    def flush(self):
        pass

    def reset_input_buffer(self):
        pass

    def reset_output_buffer(self):
        pass

    def close(self):
        pass


def parse_frames(stream):
    """按协议扫描字节流;交错/损坏的帧会抛 ValueError。"""
    i = 0
    while i < len(stream):
        b = stream[i]
        if b == 254:
            if i + 257 >= len(stream) or stream[i + 257] != 253:
                raise ValueError(f"corrupt pattern frame at byte {i}")
            yield "pattern", bytes(stream[i + 1:i + 257])
            i += 258
        elif b == 253:
            yield "commit", None
            i += 1
        else:
            raise ValueError(f"stray byte {b} at {i}")


def make_board(write_delay=0.0):
    fake = FakeSerial(write_timeout=1.0)
    with patch("serial.Serial", return_value=fake):
        board = Board("COM_FAKE")
    board.conn.write_delay = write_delay
    return board, fake


class BoardTests(unittest.TestCase):
    def test_send_encoding_unchanged(self):
        board, fake = make_board()
        board.send(phases_trap((0.0, 0.12, 0.0), maxiter=10))
        frames = list(parse_frames(fake.stream))
        self.assertEqual([k for k, _ in frames], ["pattern"])
        self.assertEqual(len(frames[0][1]), 256)
        self.assertTrue(all(0 <= b <= 32 for b in frames[0][1]))
        board.send(np.full(256, np.nan))
        frames = list(parse_frames(fake.stream))
        self.assertEqual(frames[1][1], bytes([32]) * 256)

    def test_write_timeout_retries_full_frame_once(self):
        board, fake = make_board()
        original_write = fake.write          # board.conn 就是 fake,先存原方法再打补丁
        timeouts = {"left": 1}

        def flaky_write(data):
            if timeouts["left"] > 0 and data[0] == 254:
                timeouts["left"] -= 1
                raise serial.SerialTimeoutException
            return original_write(data)

        board.conn.write = flaky_write
        board.send(my_patterns.phases_focus((0.0, 0.12, 0.0)))
        frames = list(parse_frames(fake.stream))
        self.assertEqual([k for k, _ in frames], ["pattern"])   # 整帧重发成功,无残留

    def test_concurrent_send_and_commit_never_interleave(self):
        board, fake = make_board(write_delay=0.0004)
        n_patterns, n_commits = 60, 300
        errors = []

        def sender():
            try:
                for _ in range(n_patterns):
                    board.send(my_patterns.phases_focus((0.0, 0.12, 0.0)))
            except Exception as exc:            # pragma: no cover
                errors.append(exc)

        def committer():
            try:
                for _ in range(n_commits):
                    board.send_commit()
            except Exception as exc:            # pragma: no cover
                errors.append(exc)

        t1, t2 = threading.Thread(target=sender), threading.Thread(target=committer)
        t1.start(); t2.start(); t1.join(); t2.join()
        self.assertEqual(errors, [])
        frames = list(parse_frames(fake.stream))            # 损坏帧在此抛错
        counts = {"pattern": 0, "commit": 0}
        for kind, _ in frames:
            counts[kind] += 1
        self.assertEqual(counts, {"pattern": n_patterns, "commit": n_commits})


class ModulatorTests(unittest.TestCase):
    def test_preload_pattern_updates_and_stop(self):
        board, fake = make_board()
        pat = my_patterns.phases_focus((0.0, 0.12, 0.0))
        mod = Modulator(board, pat, freq=200.0, duty=0.5).start()
        time.sleep(0.1)
        self.assertGreater(mod.cycles, 5)
        mod.set_pattern(my_patterns.phases_focus((0.01, 0.12, 0.0)))
        time.sleep(0.05)
        cycles_before_stop = mod.cycles
        mod.stop()
        self.assertIsNone(mod._thread)
        frames = list(parse_frames(fake.stream))
        kinds = [k for k, _ in frames]
        payloads = [p for k, p in frames if k == "pattern"]
        self.assertEqual(kinds[0], "pattern")            # 预装:图案
        self.assertEqual(payloads[1], bytes([32]) * 256)  # 预装:关闭图
        self.assertNotEqual(payloads[2], payloads[0])     # set_pattern 新图案
        self.assertEqual(payloads[3], bytes([32]) * 256)  # 更新后仍有关闭图
        self.assertEqual(payloads[-1], bytes([32]) * 256) # stop 的 off
        self.assertLess(mod.cycles - cycles_before_stop, 5)   # stop 及时生效
        n_after_stop = len(fake.stream)
        time.sleep(0.06)
        self.assertEqual(len(fake.stream), n_after_stop)      # 停止后不再有写入

    def test_stop_is_idempotent(self):
        board, _ = make_board()
        mod = Modulator(board, my_patterns.phases_focus((0, 0.12, 0)),
                        freq=100.0, duty=0.5).start()
        time.sleep(0.05)
        mod.stop()
        before = len(mod.board.conn.stream)
        mod.stop()
        self.assertEqual(len(mod.board.conn.stream), before)

    def test_pattern_update_never_reactivates_old_pattern(self):
        board, fake = make_board()
        old = my_patterns.phases_focus((0, 0.12, 0))
        new = my_patterns.phases_focus((0.02, 0.12, 0))
        mod = Modulator(board, old, freq=100.0, duty=0.5).start()
        time.sleep(0.025)
        mod.set_pattern(new)
        time.sleep(0.035)
        mod.stop()
        events = list(parse_frames(fake.stream))
        old_payload = next(p for kind, p in events if kind == "pattern")
        front = None
        back = None
        updated = False
        for kind, payload in events:
            if kind == "pattern":
                back = payload
                front, back = back, front
                if payload != old_payload and payload != bytes([32]) * 256:
                    updated = True
            else:
                front, back = back, front
            if updated and front is not None:
                self.assertNotEqual(front, old_payload)

    def test_commit_timeout_is_not_retried_and_modulator_stops(self):
        board, fake = make_board()
        original_write = fake.write

        def fail_commit(data):
            if data == b"\xfd":
                raise serial.SerialTimeoutException("commit timeout")
            return original_write(data)

        fake.write = fail_commit
        mod = Modulator(board, my_patterns.phases_focus((0, 0.12, 0)),
                        freq=200.0).start()
        time.sleep(0.03)
        self.assertIsInstance(mod.error, serial.SerialTimeoutException)
        mod.stop()
        self.assertEqual(list(parse_frames(fake.stream))[-1][1], bytes([32]) * 256)

    def test_hold_returns_on_eof(self):
        board, fake = make_board()
        with patch("builtins.input", side_effect=EOFError):
            hold(board, dist=0.12, freq=300.0, duty=0.5,
                 pattern=my_patterns.phases_focus((0, 0.12, 0)))
        frames = list(parse_frames(fake.stream))
        payloads = [p for k, p in frames if k == "pattern"]
        self.assertEqual(payloads[-1], bytes([32]) * 256)

    def test_burst_duration_pulse_count_and_off(self):
        board, fake = make_board()
        t0 = time.perf_counter()
        burst(board, dur=0.2, freq=200.0, duty=0.5,
              pattern=my_patterns.phases_focus((0, 0.12, 0)))
        elapsed = time.perf_counter() - t0
        self.assertGreaterEqual(elapsed, 0.19)
        self.assertLess(elapsed, 0.8)
        frames = list(parse_frames(fake.stream))
        commits = sum(1 for k, _ in frames if k == "commit")
        self.assertGreater(commits, 20)                    # 0.2s @200Hz ≈ 80
        payloads = [p for k, p in frames if k == "pattern"]
        self.assertEqual(payloads[-1], bytes([32]) * 256)


class WarmStartTests(unittest.TestCase):
    @staticmethod
    def _trap_cost(ph, focus):
        dy, ring_r, n_ring, w = 0.035, 0.03, 10, 60.0
        pts = [focus]
        for h in (focus[1] - dy, focus[1] + dy):
            for a in np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False):
                pts.append([focus[0] + ring_r * np.cos(a), h, focus[2] + ring_r * np.sin(a)])
        f = my_patterns._propagators(np.array(pts)).T @ np.exp(1j * ph)
        return -np.abs(f[0]) ** 2 + w * np.sum(np.abs(f[1:]) ** 2)

    def test_trap_warm_start_no_worse_than_cold(self):
        f1, f2 = (0.0, 0.12, 0.0), (0.005, 0.12, 0.0)
        prev = phases_trap(f1, maxiter=500)
        cold = phases_trap(f2, maxiter=500)
        warm = phases_trap(f2, maxiter=500, x0=prev)
        cost_cold = self._trap_cost(cold, f2)
        cost_warm = self._trap_cost(warm, f2)
        self.assertLessEqual(cost_warm, cost_cold + 0.1 * abs(cost_cold) + 1e-9)
        self.assertLess(cost_warm, self._trap_cost(my_patterns.phases_focus(f2), f2))

    def test_pattern_for_kind_passes_x0_only_to_trap(self):
        ph1 = pattern_for_kind("trap", 0.0, 0.0, 0.12)
        ph2 = pattern_for_kind("trap", 0.005, 0.0, 0.12, x0=ph1)
        self.assertEqual(ph2.shape, (256,))
        # 非 trap 类型不受 x0 影响
        self.assertTrue(np.allclose(pattern_for_kind("focus", 0.01, 0.0, 0.12),
                                    my_patterns.phases_focus((0.01, 0.12, 0.0))))

    def test_gs_warm_start_continues_trajectory(self):
        pts = np.stack(np.meshgrid(np.linspace(-0.05, 0.05, 8),
                                   np.full(8, 0.12),
                                   np.linspace(-0.05, 0.05, 8)), axis=-1).reshape(-1, 3)
        G = my_patterns._propagators(pts)
        amps = 0.5 * np.ones(len(pts))
        a = my_patterns._gs_from_model(G, amps, 10, 0)
        b = my_patterns._gs_from_model(G, amps, 10, 0, x0=a)   # 从 a 温启动再跑 10 轮
        c = my_patterns._gs_from_model(G, amps, 20, 0)         # 同种子连续跑 20 轮
        self.assertEqual(b.shape, (256,))
        self.assertTrue(np.all(np.isfinite(b)))
        np.testing.assert_allclose(b, c, atol=1e-6)            # 温启动=轨迹精确延续

    def test_gs_default_starts_from_target_backprop(self):
        rng = np.random.default_rng(4)
        G = rng.normal(size=(256, 8)) + 1j * rng.normal(size=(256, 8))
        amps = rng.uniform(0, 1, 8)
        expected = my_patterns._wrap(np.angle(np.conj(G @ amps)))
        actual = my_patterns._gs_from_model(G, amps, 0, 0)
        np.testing.assert_allclose(actual, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
