from src.simulation.run_orbit_simulation import run_orbit_simulation
from src.utils.quick_plot import plot_results

result = run_orbit_simulation()
plot_results(result)
