import rdkit


def get_atom_props(atom: rdkit.Chem.rdchem.Atom, ring_size: int = 1) -> tuple:
    """
    Get properties of atom

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        Atom
    ring_size : int
        The maximal ring size to look for

    Returns
    -------
    atom_props : tuple
        Tuple of atom's properties
    """
    return tuple([
        atom.GetSymbol(),
        atom.GetDegree(),
        atom.GetTotalDegree(),
        atom.GetImplicitValence(),
        atom.GetExplicitValence(),
        atom.GetTotalValence(),
        atom.GetNumImplicitHs(),
        atom.GetNumExplicitHs(),
        atom.GetTotalNumHs(),
        atom.GetIsAromatic(),
        atom.IsInRing(),
        *[atom.IsInRingSize(i) for i in range(ring_size)]
    ])


def get_bond_props(bond: rdkit.Chem.rdchem.Bond,
                   atom: rdkit.Chem.rdchem.Atom,
                   atoms_to_sym_cls: dict) -> tuple:
    """
    Get properties of bond

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        Bond
    atom : rdkit.Chem.rdchem.Atom
        Bond's atom
    atoms_to_sym_cls : dict { atom_idx (int) -> sym_class_idx (int) }
        Table of atoms symmetry classes

    Returns
    -------
    bond_props : tuple
        Tuple of bond's properties
    """
    return tuple([
        int(bond.GetBondDir()),
        bond.GetBondTypeAsDouble(),
        int(bond.GetIsAromatic()),
        int(bond.GetIsConjugated()),
        atoms_to_sym_cls[bond.GetOtherAtom(atom).GetIdx()]
    ])


def get_atom_bonds_props(atom: rdkit.Chem.rdchem.Atom, atoms_to_sym_cls: dict) -> tuple:
    """
    Get bonds properties for atom

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        Atom
    atoms_to_sym_cls : dict { atom_idx (int) -> sym_class_idx (int) }
        Table of atoms symmetry classes

    Returns
    -------
    bonds_props : tuple
        Sorted tuple of bonds properties for atom
    """
    bonds_props = [
        get_bond_props(bond, atom, atoms_to_sym_cls) for bond in atom.GetBonds()
    ]
    if len(bonds_props) > 1:
        # Sort bonds properties in alphabetical order
        for i in range(len(bonds_props[0])):
            bonds_props.sort(key=lambda e: e[i])
    return tuple(bonds_props)


def find_mol_sym_atoms(mol: rdkit.Chem.rdchem.Mol) -> list:
    """
    Find symmetric atoms in molecule

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to find symmetric atoms

    Returns
    -------
    sym_atoms : list of sets of integers
        List of sets of symmetric atoms
    """
    sym_atoms = []

    # Step 1: Initialize list of sets of symmetric atoms using atoms properties only
    atoms_props = {}
    for atom in mol.GetAtoms():
        atom_key = get_atom_props(atom, ring_size=mol.GetNumAtoms())
        if atom_key in atoms_props:
            atoms_props[atom_key].add(atom)
        else:
            atoms_props[atom_key] = set({atom})
    sym_atoms = list(atoms_props.values())
    atoms_to_sym_cls = {
        atom.GetIdx(): i for i, cls in enumerate(sym_atoms) for atom in cls
    }

    # Step 2: Cut sets of symmetric atoms until atoms have different bonds properties
    while True:
        is_break = True
        for cls_idx, cls in enumerate(sym_atoms):
            if len(cls) > 1:
                cls_atoms_props = {}
                for atom in cls:
                    atom_key = get_atom_bonds_props(atom, atoms_to_sym_cls)
                    if atom_key in cls_atoms_props:
                        cls_atoms_props[atom_key].add(atom)
                    else:
                        cls_atoms_props[atom_key] = set({atom})
                if len(cls_atoms_props) > 1:
                    cls_sym_atoms = list(cls_atoms_props.values())
                    for i in range(1, len(cls_sym_atoms)):
                        subcls = cls_sym_atoms[i]
                        subcls_idx = len(sym_atoms)
                        cls = cls - subcls
                        sym_atoms.append(subcls)
                        for atom in subcls:
                            atoms_to_sym_cls[atom.GetIdx()] = subcls_idx
                    sym_atoms[cls_idx] = cls
                    is_break = False
        if is_break:
            break

    return [set({atom.GetIdx() for atom in cls}) for cls in sym_atoms]
