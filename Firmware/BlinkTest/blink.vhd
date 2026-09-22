-- Minimal LED blink test for SonicSurface (EP4CE6/10CL, 50 MHz clock)
-- Blinks all 4 array LEDs (L1-L4) at 1 Hz
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity blink is
    port (
        CLK : in  std_logic;                    -- 50 MHz oscillator (PIN_23)
        LED : out std_logic_vector(3 downto 0)  -- L1..L4 on array PCB
    );
end blink;

architecture rtl of blink is
    signal counter    : unsigned(24 downto 0) := (others => '0');
    signal led_state  : std_logic := '0';
begin
    process(CLK)
    begin
        if rising_edge(CLK) then
            if counter = 24_999_999 then   -- 0.5 s at 50 MHz
                counter   <= (others => '0');
                led_state <= not led_state;
            else
                counter <= counter + 1;
            end if;
        end if;
    end process;

    LED <= (others => led_state);
end rtl;
