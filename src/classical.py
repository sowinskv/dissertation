"""
classical: reference energies from PySCF (HF, CCSD, FCI).

FCI is the exact answer in the basis and the yardstick for every error in the thesis.
CCSD is not variational: with strong correlation it can land BELOW the exact energy.
"""

import numpy as np
from pyscf import cc, fci, gto, scf

from src.hamiltonians import BASIS


def reference_energies(atom, basis=BASIS):
    """HF, CCSD and FCI total energies (Ha) plus the weight of the HF determinant in the FCI state.

    hf_weight = |<HF|FCI>|^2 : close to 1 = one configuration dominates (weak correlation),
    far below 1 = several configurations matter (strong correlation, HF and CCSD struggle).
    """
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    e_fci, ci = fci.FCI(mf).kernel()
    ccsd = cc.CCSD(mf).run()
    return {
        "E_HF_Ha": float(mf.e_tot),
        "E_CCSD_Ha": float(ccsd.e_tot) if ccsd.converged else np.nan,
        "E_FCI_Ha": float(e_fci),
        "hf_weight": float(np.abs(np.asarray(ci)[0, 0]) ** 2),
    }
