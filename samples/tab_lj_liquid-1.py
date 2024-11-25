"""
Simulate a Lennard-Jones fluid maintained at a fixed temperature
by a Langevin thermostat. Shows the basic features of how to:

* set up system parameters, particles and interactions.
* warm up and integrate.
* write parameters, configurations and observables to files.

The particles in the system are of two types: type 0 and type 1.
Type 0 particles interact with each other via a repulsive WCA
interaction. Type 1 particles neither interact with themselves
nor with type 0 particles.
"""
import numpy as np
import espressomd

required_features = ["LENNARD_JONES"]
espressomd.assert_features(required_features)

print("""
=======================================================
=                    lj_liquid.py                     =
=======================================================
""")

# System parameters
#############################################################

box_l = 10.7437
density = 0.7

# Interaction parameters (repulsive Lennard-Jones)
#############################################################

lj_eps = 1.0
lj_sig = 1.0
lj_cut = 2.5 * lj_sig

# Integration parameters
#############################################################
system = espressomd.System(box_l=[box_l] * 3)
np.random.seed(seed=42)

system.time_step = 0.01
system.cell_system.skin = 0.4

# warmup integration (steepest descent)
warm_steps = 20
warm_n_times = 10
# convergence criterion (particles are separated by at least 90% sigma)
min_dist = 0.9 * lj_sig

# integration
int_steps = 1000
int_n_times = 5


#############################################################
#  Setup System                                             #
#############################################################

e_tab, f_tab = system.non_bonded_inter[0,0].tabulated.get_table(sig=lj_sig,eps=lj_eps,f="4*eps*((sig/r)**12-(sig/r)**6)",min=0,max=lj_cut,steps=1000)


# Interaction setup
#############################################################
system.non_bonded_inter[0, 0].tabulated.set_params(min=0.0, max=lj_cut,energy=e_tab, force=f_tab)

print("LJ-parameters:")
#print(system.non_bonded_inter[0, 0].tabulated.get_params())

# Particle setup
#############################################################

volume = box_l**3
n_part = int(volume * density)

for i in range(n_part):
    system.part.add(pos=np.random.random(3) * system.box_l)

print(
    f"Simulate {n_part} particles in a cubic box of length {box_l} at density {density}.")
print("Interactions:\n")
act_min_dist = system.analysis.min_dist()
print(f"Start with minimal distance {act_min_dist}")


#############################################################
#  Warmup Integration                                       #
#############################################################

print(f"""\
Start warmup integration:
At maximum {warm_n_times} times {warm_steps} steps
Stop if minimal distance is larger than {min_dist}""")
#print(system.non_bonded_inter[0, 0].tabulated)

# minimize energy using min_dist as the convergence criterion
system.integrator.set_steepest_descent(f_max=0, gamma=1e-3,
                                       max_displacement=lj_sig / 100)
i = 0
while i < warm_n_times and system.analysis.min_dist() < min_dist:
    print(f"minimization: {system.analysis.energy()['total']:+.2e}")
    system.integrator.run(warm_steps)
    i += 1

print(f"minimization: {system.analysis.energy()['total']:+.2e}")
print()
system.integrator.set_vv()

# activate thermostat
system.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=42)

# Just to see what else we may get from the C++ core
import pprint
pprint.pprint(system.cell_system.get_state(), width=1)
# pprint.pprint(system.part.__getstate__(), width=1)
pprint.pprint(system.__getstate__())


#############################################################
#      Integration                                          #
#############################################################
print(f"\nStart integration: run {int_n_times} times {int_steps} steps")

for i in range(int_n_times):
    print(f"run {i} at time={system.time:.2f}")

    system.integrator.run(steps=int_steps)

    energies = system.analysis.energy()
    print(energies['total'])
    linear_momentum = system.analysis.linear_momentum()
    print(linear_momentum)


# terminate program
print("\nFinished.")
