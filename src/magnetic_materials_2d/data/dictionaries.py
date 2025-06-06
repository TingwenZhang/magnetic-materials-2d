# flake8: noqa
"""A collection of dictionaries for the dataset."""

column_meaning_map = {
    "formation_energy": "Formation energy [eV]",
    "elem_frac": "Relative fraction of elements in the structure [dimensionless]",
    "elem_list": "List of elements in the structure [dimensionless]",
    "energy": "DFT total energy [eV]",
    "formula": "Formula [dimensionless]",
    "magnetic_moment": "Magnetic moment per unit cell [\mathrm{\mu_B}]",
    "spin_index": "Spin index: 0 - FM, 1 - AFM [dimensionless]",
    "num_p": "Number of p electrons [count]",
    "num_d": "Number of d electrons [count]",
    "num_f": "Number of f electrons [count]",
    "atomic_rad": "Atomic radius of the atoms in the structure [\mathrm{\AA}]",
    "atomic_vol": "Atomic volume of the atoms in the structure [\mathrm{\AA}^3]",
    "covalent_rad": "Covalent radius of the atoms in the structure [\mathrm{\AA}]",
    "dipole": "Dipole polarizability of the atoms in the structure [\mathrm{\AA}^3]",
    "eaffinity": "Electron affinity of the atoms in the structure [eV]",
    "num_electrons": "Number of electrons [count]",
    "atomic_rad_sum_dif": "Sum of the differences of atomic radii [\mathrm{\AA}]",
    "atomic_rad_std_dif": "Standard deviation of differences of atomic radii [\mathrm{\AA}]",
    "atomic_rad_std": "Standard deviation of atomic radii [\mathrm{\AA}]",
    "atomic_rad_avg": "Mean of the atomic radii [\mathrm{\AA}]",
    "atomic_rad_max_dif": "Maximum difference of atomic radii [\mathrm{\AA}]",
    "atomic_vol_sum_dif": "Sum of differences of atomic volumes [\mathrm{\AA}^3]",
    "atomic_vol_std_dif": "Standard deviation of differences of atomic volumes [\mathrm{\AA}^3]",
    "atomic_vol_std": "Standard deviation of atomic volumes [\mathrm{\AA}^3]",
    "atomic_vol_avg": "Mean of atomic volumes [\mathrm{\AA}^3]",
    "atomic_vol_max_dif": "Maximum difference of atomic volumes [\mathrm{\AA}^3]",
    "covalentrad_sum_dif": "Sum of differences of covalent radii [\mathrm{\AA}]",
    "covalentrad_std_dif": "Standard deviation of covalent radii differences [\mathrm{\AA}]",
    "covalentrad_std": "Standard deviation of covalent radii [\mathrm{\AA}]",
    "covalentrad_avg": "Mean of covalent radii [\mathrm{\AA}]",
    "covalentrad_max_dif": "Maximum difference of covalent radii [\mathrm{\AA}]",
    "dipole_sum_dif": "Sum of dipole polarizability differences [\mathrm{\AA}^3]",
    "dipole_std_dif": "Standard deviation of dipole polarizability differences [\mathrm{\AA}^3]",
    "dipole_std": "Standard deviation of dipole polarizability [\mathrm{\AA}^3]",
    "dipole_avg": "Mean of dipole polarizability [\mathrm{\AA}^3]",
    "dipole_max_dif": "Maximum difference of dipole polarizability [\mathrm{\AA}^3]",
    "eaffinity_sum_dif": "Sum of electron affinity differences [eV]",
    "eaffinity_std_dif": "Standard deviation of electron affinity differences [eV]",
    "eaffinity_std": "Standard deviation of electron affinity [eV]",
    "e_affinity_avg": "Mean of electron affinity [eV]",
    "e_affinity_max_dif": "Maximum difference of electron affinity [eV]",
    "numelectron_sum_dif": "Sum of electron count differences [count]",
    "numelectron_std_dif": "Standard deviation of electron count differences [count]",
    "numelectron_std": "Standard deviation of electron counts [count]",
    "numelectron_avg": "Mean of electron counts [count]",
    "numelectron_max_dif": "Maximum difference of electron counts [count]",
    "vdwradius_sum_dif": "Sum of van der Waals radii differences [\mathrm{\AA}]",
    "vdwradius_std_dif": "Standard deviation of van der Waals radii differences [\mathrm{\AA}]",
    "vdwradius_std": "Standard deviation of van der Waals radii [\mathrm{\AA}]",
    "vdwradius_avg": "Mean of van der Waals radii [\mathrm{\AA}]",
    "vdwradius_max_dif": "Maximum difference of van der Waals radii [\mathrm{\AA}]",
    "e_negativity_sum_dif": "Sum of electronegativity differences [Pauling]",
    "e_negativity_std_dif": "Standard deviation of electronegativity differences [Pauling]",
    "e_negativity_std": "Standard deviation of electronegativity [Pauling]",
    "e_negativity_avg": "Mean of electronegativity [Pauling]",
    "e_negativity_max_dif": "Maximum electronegativity difference [Pauling]",
    "nvalence_sum_dif": "Sum of valence electron differences [count]",
    "nvalence_std_dif": "Standard deviation of valence electron differences [count]",
    "nvalence_std": "Standard deviation of valence electrons [count]",
    "nvalence_avg": "Mean of valence electrons [count]",
    "nvalence_max_dif": "Maximum valence electron difference [count]",
    "lastsubshell_avg": "Mean electrons in last subshell [count]",
    "cmpd_skew_p": "Skew of p electrons [dimensionless]",
    "cmpd_skew_d": "Skew of d electrons [dimensionless]",
    "cmpd_skew_f": "Skew of f electrons [dimensionless]",
    "cmpd_sigma_p": "Standard deviation of p electrons [count]",
    "cmpd_sigma_d": "Standard deviation of d electrons [count]",
    "cmpd_sigma_f": "Standard deviation of f electrons [count]",
    "frac_f ": "Fraction of f electrons [dimensionless]",
    "std_ion": "Standard deviation of ionization energies [eV]",
    "sum_ion": "Sum of ionization energies [eV]",
    "mean_ion": "Mean ionization energy [eV]",
    "Born": "Born-Haber term [not sure]",
    "hardness_mean": "Mean chemical hardness [eV]",
    "hardness_var": "Variance of chemical hardness [\mathrm{eV}^2]",
    "Nup_mean": "Mean unpaired electrons [count]",
    "Nup_var": "Variance of unpaired electrons [count$^2$]",
    "cs_bob": "Bag of bonds chemical space [dimensionless]",
    "cs_PE": "Pauling electronegativity BoB flavor [dimensionless]",
    "cs_IR": "Ionic radius BoB flavor [dimensionless]",
    "cs_AR": "Atomic radius BoB flavor [dimensionless]",
    "cs_OX": "Oxidation number BoB flavor [dimensionless]",
}


formation_energy_map = {"unit": "[eV / cell]", "label": "formation_energy"}


magnetic_moment_map = {
    "unit": "[Bohr magneton / cell]",
    "label": "magnetic_moment",
}
