from .fp_calculator import set_sym, cal_fp_only, conditional_cal_fp_only
import torch
import numpy as np
from ase import Atoms
from ase.units import *


def conditional_cal_nrg_local(models, atoms, params_set, elements, scale, conditions):
    # only calculate the related atomic energies, faster
	fp_data = conditional_cal_fp_only(atoms, elements, params_set, conditions)
	fps = [fp_data['x'][el] for el in elements]  # fp for each element in order
	fps = [torch.FloatTensor(fp) for fp in fps]
	fps = [(fp - scale['fp_min'])/(scale['fp_max'] - scale['fp_min']) for fp in fps]
	res = 0
	for i in range(len(models)):
		if len(fps[i]) != 0:
			res += torch.sum(models[i](fps[i])).item()
	return res


def cal_nrg(models, atoms, params_set, elements, scale):
    fp_data = cal_fp_only(atoms, elements, params_set)
    fps = [fp_data['x'][el] for el in elements]  # fp for each element in order
    fps = [torch.FloatTensor(fp) for fp in fps]
    fps = [(fp - scale['fp_min'])/(scale['fp_max'] - scale['fp_min']) for fp in fps]
    res = 0
    for i in range(len(models)):
    	if len(fps[i]) != 0:
    		res += torch.sum(models[i](fps[i])).item()
    return res


def create_single_cluster(atoms, nl, ind, Rc):
    positions = atoms.positions
    center_ps = positions[ind]
    syms = np.asarray(atoms.get_chemical_symbols())
    cells = atoms.cell
    indices, offsets = nl.get_neighbors(ind)
    indices = np.append(indices, ind)
    offsets = np.vstack((offsets, [0, 0, 0]))
    nb_positions = positions[indices] + offsets @ cells 
    nb_syms = syms[indices]
    cluster = Atoms(nb_syms, nb_positions)
    cluster.center(vacuum=Rc+0.5)
    conditions = np.linalg.norm(nb_positions-center_ps, axis=1) < Rc
    return cluster, conditions


def create_clusters(atoms, nl, inds, Rc):
    center_ps_1 = atoms.positions[inds[0]]
    center_ps_2 = atoms.positions[inds[1]]
    
    positions = atoms.positions
    syms = np.asarray(atoms.get_chemical_symbols())
    cells = atoms.cell

    # operate the first central atom
    ind_1, ind_2 = inds
    indices_1, offsets_1 = nl.get_neighbors(ind_1)
    indices_1 = np.append(indices_1, ind_1)
    offsets_1 = np.vstack((offsets_1, [0, 0, 0]))
    nb_1_positions = positions[indices_1] + offsets_1 @ cells 
    nb_1_syms = syms[indices_1]
    ind_2_offset = [0, 0, 0]
    if ind_2 in indices_1:
        ind_2_cond = (indices_1 == ind_2)
        ind_2_ps = nb_1_positions[ind_2_cond]
        ind_2_offsets = offsets_1[ind_2_cond]
        ind_2_ds = np.linalg.norm(ind_2_ps - center_ps_1, axis=1)
        flag = np.argmin(ind_2_ds)
        if ind_2_ds[flag] < np.linalg.norm(center_ps_1 - center_ps_2):
            center_ps_2 = ind_2_ps[flag]
            ind_2_offset = ind_2_offsets[flag]

    indices_2, offsets_2 = nl.get_neighbors(ind_2)
    indices_2 = np.append(indices_2, ind_2)
    offsets_2 = np.vstack((offsets_2, [0, 0, 0]))
    nb_2_positions = positions[indices_2] + offsets_2 @ cells + ind_2_offset @ cells[None, :]
    nb_2_syms = syms[indices_2]

    flags = np.linalg.norm(nb_1_positions - center_ps_2[None, :], axis=1) >= 2*Rc 
    tmp_positions = nb_1_positions[flags]
    tmp_syms = nb_1_syms[flags]
    all_positions = np.vstack([tmp_positions, nb_2_positions])
    all_syms = np.append(tmp_syms, nb_2_syms)

    cluster = Atoms(all_syms, all_positions)
    cluster.center(vacuum=Rc+0.5)

    condition_tmp = (np.linalg.norm(all_positions-center_ps_1, axis=1) < Rc) | (np.linalg.norm(all_positions-center_ps_2, axis=1) < Rc)
    conditions = condition_tmp

    return cluster, conditions


class Surface_Adsorp_Sites():
    def __init__(self, slab, a, ad_nrgs_dict, P, T, lc, m):
        """
        slab: Atoms, slab
        a: int, length of slab square
        lc: float, lattice constant
        ad_nrg_dict: dict, dict of adsorption energies, eV/site
        m: mass of adsorbate molecule, kg/mol
        P: float, adsorbate pressure, Pa
        T: float, temperature, K
        """
        self.a = a
        self.slab = slab
        self.adsorp_sites, self.atom_adsorp_sites_inds = self.get_adsorb_sites(self.slab, self.a)
        self.ad_nrgs_dict = ad_nrgs_dict
        cov_dict = dict()
        for k in self.ad_nrgs_dict.keys():
            ad_nrg = self.ad_nrgs_dict[k]
            cov = self.cal_cov(ad_nrg, P, T, lc, m)
            cov_dict[k] = cov
        self.cov_dict = cov_dict

    def get_adsorb_sites(self, slab, a):
        n_total = len(slab)
        fcc_inds = []
        surf_inds = np.append(np.arange(0, a**2), np.arange(n_total-a**2, n_total))

        # adsorption site: [[indices of atoms that form the adsorption site], [cooresponding elements]]
        # for sake of simpilicity, we only consider the fcc sites
        # one surface
        for i in range(a**2-a):
            if i % a != 0:
                fcc_inds.append([[i, i+a-1, i+a], [slab[i].symbol, slab[i+a-1].symbol, slab[i+a].symbol]])
            else:
                fcc_inds.append([[i, i+2*a-1, i+a], [slab[i].symbol, slab[i+2*a-1].symbol, slab[i+a].symbol]])
        
        for i in range(a**2-a, a**2):
            if i % a == 0:
                fcc_inds.append([[i, i%a, (i%a)+a-1], [slab[i].symbol, slab[i%a].symbol, slab[(i%a)+a-1].symbol]])
            else:
                fcc_inds.append([[i, i%a, (i%a)-1], [slab[i].symbol, slab[i%a].symbol, slab[(i%a)-1].symbol]])

        # the other surface
        for i in range(n_total-a**2, n_total-a):
            if i % a != (a-1):
                fcc_inds.append([[i, i+1, i+a], [slab[i].symbol, slab[i+1].symbol, slab[i+a].symbol]])
            else:
                fcc_inds.append([[i, i+a, i-a+1], [slab[i].symbol, slab[i+a].symbol, slab[i-a+1].symbol]])

        for i in range(n_total-a, n_total):
            if i % a != (a-1):
                fcc_inds.append([[i, i+1, i-a**2+a], [slab[i].symbol, slab[i+1].symbol, slab[i-a**2+a].symbol]])
            else:
                fcc_inds.append([[i, i-a+1, i-a**2+a], [slab[i].symbol, slab[i-a+1].symbol, slab[i-a**2+a].symbol]])

        sites = fcc_inds
        atom_sites_inds = dict()
        for i in surf_inds:
            atom_sites_inds[i] = []
        
        for i in range(len(sites)):
            site = sites[i]
            for ind in site[0]:
                atom_sites_inds[ind].append(i)

        return sites, atom_sites_inds

    def get_ad_nrg_cov(self, site):
        # number of Pd in each site determines the adsorption energy of this site
        site_config = site[-1]
        n_Pd = site_config.count('Pd')
        return self.cov_dict[n_Pd], self.ad_nrgs_dict[n_Pd]

    def cal_cov(self, ad_nrg, P, T, lc, m):
        Na = mol  # Avocadoro constant
        A = np.sqrt(3)/4*lc**2  # area of each adsorption site 
        S = 10**20/A  # adsorption site density, #/m**2
        v = 10**13  # pre exponential constant of desorption rate
        pi = np.pi
        m = 0.032
        R = 8.314
        des_nrg = -ad_nrg
        K = Na/(S*v) * np.sqrt(1/(2*pi*m*R*T))*np.exp(des_nrg/(kB*T))
        K *= 2  # one oxygen molecule to 2 oxygen atoms
        Keq = K**2  # convert to adapt O2 molecule
        cov = np.sqrt(Keq*P)/(1+np.sqrt(Keq*P))
        return cov

    def update_sites(self, slab, sites):
        self.slab = slab
        self.adsorp_sites = sites


def delta_mu_T(T):
    """
    return as eV/molecule
    """
    K = 1
    A = 31.32234; B = -20.23532; C = 57.86644
    D = -36.50624; E = -0.007374; F = -8.903471
    G = 246.7945; H = 0.0
    t = T/1000
    # entropy
    s = (A*np.log(t) + B*t + C*(t**2)/2. 
        + D*(t**3)/3. - E/(2. * t**2) + G)
    s = s*J/mol/K
    # enthalpy
    h = (A*t + B*(t**2)/2. + C*(t**3)/3. 
        + D*(t**4)/4. - E/t + F-H)
    h = h*kJ/mol
    return h - T*s


class Atom_Adsorp_Sites():
    def __init__(self, slab, a, ad_nrgs_dict):
        """
        slab: Atoms, slab
        a: int, length of slab square
        ad_nrg_dict: dict, dict of adsorption energies, eV/site
        """
        self.a = a
        self.slab = slab
        self.adsorp_sites, self.atom_adsorp_sites_inds = self.get_adsorb_sites(self.slab, self.a)
        self.ad_nrgs_dict = ad_nrgs_dict

    def get_adsorb_sites(self, slab, a):
        n_total = len(slab)
        fcc_inds = []
        surf_inds = np.append(np.arange(0, a**2), np.arange(n_total-a**2, n_total))

        # adsorption site: [[indices of atoms that form the adsorption site], [cooresponding elements], is_occupied]
        # for sake of simpilicity, we only consider the fcc sites
        # one surface
        for i in range(a**2-a):
            if i % a != 0:
                fcc_inds.append([[i, i+a-1, i+a], [slab[i].symbol, slab[i+a-1].symbol, slab[i+a].symbol], False])
            else:
                fcc_inds.append([[i, i+2*a-1, i+a], [slab[i].symbol, slab[i+2*a-1].symbol, slab[i+a].symbol], False])
        
        for i in range(a**2-a, a**2):
            if i % a == 0:
                fcc_inds.append([[i, i%a, (i%a)+a-1], [slab[i].symbol, slab[i%a].symbol, slab[(i%a)+a-1].symbol], False])
            else:
                fcc_inds.append([[i, i%a, (i%a)-1], [slab[i].symbol, slab[i%a].symbol, slab[(i%a)-1].symbol], False])

        # the other surface
        for i in range(n_total-a**2, n_total-a):
            if i % a != (a-1):
                fcc_inds.append([[i, i+1, i+a], [slab[i].symbol, slab[i+1].symbol, slab[i+a].symbol], False])
            else:
                fcc_inds.append([[i, i+a, i-a+1], [slab[i].symbol, slab[i+a].symbol, slab[i-a+1].symbol], False])

        for i in range(n_total-a, n_total):
            if i % a != (a-1):
                fcc_inds.append([[i, i+1, i-a**2+a], [slab[i].symbol, slab[i+1].symbol, slab[i-a**2+a].symbol], False])
            else:
                fcc_inds.append([[i, i-a+1, i-a**2+a], [slab[i].symbol, slab[i-a+1].symbol, slab[i-a**2+a].symbol], False])

        sites = fcc_inds
        atom_sites_inds = dict()
        for i in surf_inds:
            atom_sites_inds[i] = []
        
        for i in range(len(sites)):
            site = sites[i]
            for ind in site[0]:
                atom_sites_inds[ind].append(i)

        return sites, atom_sites_inds

    def get_ad_nrg(self, site):
        # number of Pd in each site determines the adsorption energy of this site
        site_config = site[1]
        n_Pd = site_config.count('Pd')
        return self.ad_nrgs_dict[n_Pd]

    def update_sites(self, slab, sites):
        self.slab = slab
        self.adsorp_sites = sites
        
    def test_site_ad(self, site, T):  # for empty site, test if adsorb
        ad_nrg = self.get_ad_nrg(site)
        flag = False
        if ad_nrg < 0:
            flag = True
        elif np.exp(-ad_nrg / (kB * T)) > np.random.rand():
            flag = True
        return flag  # if occupy this empty site

    def test_site_des(self, site, T):  # for occupied site, test if desorb
        des_nrg = -self.get_ad_nrg(site)
        flag = False
        if des_nrg < 0:
            flag = True
        elif np.exp(-des_nrg / (kB * T)) > np.random.rand():
            flag = True
        return flag  # if desorb from this occupied site

    def update_adsorbates(self, T):
        for i in range(len(self.adsorp_sites)):
            site = self.adsorp_sites[i]
            if site[-1]:  # if occupied
                if self.test_site_des(site, T):
                    self.adsorp_sites[i][-1] = False  # desorb
            else:  # if empty
                if self.test_site_ad(site, T):
                    self.adsorp_sites[i][-1] = True  # adsorb
