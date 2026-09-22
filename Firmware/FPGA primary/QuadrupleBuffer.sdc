# Constraints for QuadrupleBuffer (SonicSurface primary board)
# Input clock: 50 MHz on pin CLK (PIN_23)
# PLL (Masterclock/ALTPLL): c0 = 50MHz * 128/625 = 10.24 MHz phase-step clock

create_clock -period 20.000 -name clk50 [get_ports {CLK}]
derive_pll_clocks
derive_clock_uncertainty
