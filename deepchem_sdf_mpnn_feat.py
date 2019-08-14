class WeaveMol(object):
  """Holds information about a molecule
  Molecule struct used in weave models
  """

  def __init__(self, nodes, pairs):
    self.nodes = nodes
    self.pairs = pairs
    self.num_atoms = self.nodes.shape[0]
    self.n_features = self.nodes.shape[1]

  def get_pair_features(self):
    return self.pairs

  def get_atom_features(self):
    return self.nodes

  def get_num_atoms(self):
    return self.num_atoms

  def get_num_features(self):
    return self.n_features


class SDFLoader(DataLoader):
  """
  Handles loading of SDF files.
  """

  def __init__(self, tasks, clean_mols=False, **kwargs):
    super(SDFLoader, self).__init__(tasks, **kwargs)
    self.clean_mols = clean_mols
    self.tasks = tasks
    self.smiles_field = "smiles"
    self.mol_field = "mol"
    self.id_field = "smiles"

  def get_shards(self, input_files, shard_size):
    """Defines a generator which returns data for each shard"""
    return load_sdf_files(input_files, self.clean_mols, tasks=self.tasks)

  def featurize_shard(self, shard):
    """Featurizes a shard of an input dataframe."""
    log(
        "Currently featurizing feature_type: %s" %
        self.featurizer.__class__.__name__, self.verbose)
    return featurize_mol_df(shard, self.featurizer, field=self.mol_field)


class Featurizer(object):
  """
  Abstract class for calculating a set of features for a molecule.
  Child classes implement the _featurize method for calculating features
  for a single molecule.
  """

  def featurize(self, mols, verbose=True, log_every_n=1000):
    """
    Calculate features for molecules.
    Parameters
    ----------
    mols : iterable
        RDKit Mol objects.
    """
    mols = list(mols)
    features = []
    for i, mol in enumerate(mols):
      if mol is not None:
        features.append(self._featurize(mol))
      else:
        features.append(np.array([]))

    features = np.asarray(features)
    return features

class WeaveFeaturizer(Featurizer):
  name = ['weave_mol']

  def __init__(self, graph_distance=True, explicit_H=False,
               use_chirality=False):
    # Distance is either graph distance(True) or Euclidean distance(False,
    # only support datasets providing Cartesian coordinates)
    self.graph_distance = graph_distance
    # Set dtype
    self.dtype = object
    # If includes explicit hydrogens
    self.explicit_H = explicit_H
    # If uses use_chirality
    self.use_chirality = use_chirality

  def _featurize(self, mol):
    """Encodes mol as a WeaveMol object."""
    # Atom features
    idx_nodes = [(a.GetIdx(),
                  atom_features(
                      a,
                      explicit_H=self.explicit_H,
                      use_chirality=self.use_chirality))
                 for a in mol.GetAtoms()]
    idx_nodes.sort()  # Sort by ind to ensure same order as rd_kit
    idx, nodes = list(zip(*idx_nodes))

    # Stack nodes into an array
    nodes = np.vstack(nodes)

    # Get bond lists
    edge_list = {}
    for b in mol.GetBonds():
      edge_list[tuple(sorted([b.GetBeginAtomIdx(),
                              b.GetEndAtomIdx()]))] = bond_features(
                                  b, use_chirality=self.use_chirality)

    # Get canonical adjacency list
    canon_adj_list = [[] for mol_id in range(len(nodes))]
    for edge in edge_list.keys():
      canon_adj_list[edge[0]].append(edge[1])
      canon_adj_list[edge[1]].append(edge[0])

    # Calculate pair features
    pairs = pair_features(
        mol,
        edge_list,
        canon_adj_list,
        bt_len=6,
        graph_distance=self.graph_distance)

    return WeaveMol(nodes, pairs)

def pair_features(mol, edge_list, canon_adj_list, bt_len=6,
                  graph_distance=True):
  if graph_distance:
    max_distance = 7
  else:
    max_distance = 1
  N = mol.GetNumAtoms()
  features = np.zeros((N, N, bt_len + max_distance + 1))
  num_atoms = mol.GetNumAtoms()
  rings = mol.GetRingInfo().AtomRings()
  for a1 in range(num_atoms):
    for a2 in canon_adj_list[a1]:
      # first `bt_len` features are bond features(if applicable)
      features[a1, a2, :bt_len] = np.asarray(
          edge_list[tuple(sorted((a1, a2)))], dtype=float)
    for ring in rings:
      if a1 in ring:
        # `bt_len`-th feature is if the pair of atoms are in the same ring
        features[a1, ring, bt_len] = 1
        features[a1, a1, bt_len] = 0.
    # graph distance between two atoms
    if graph_distance:
      distance = find_distance(
          a1, num_atoms, canon_adj_list, max_distance=max_distance)
      features[a1, :, bt_len + 1:] = distance
  # Euclidean distance between atoms
  if not graph_distance:
    coords = np.zeros((N, 3))
    for atom in range(N):
      pos = mol.GetConformer(0).GetAtomPosition(atom)
      coords[atom, :] = [pos.x, pos.y, pos.z]
    features[:, :, -1] = np.sqrt(np.sum(np.square(
      np.stack([coords] * N, axis=1) - \
      np.stack([coords] * N, axis=0)), axis=2))

  return features

def find_distance(a1, num_atoms, canon_adj_list, max_distance=7):
  distance = np.zeros((num_atoms, max_distance))
  radial = 0
  # atoms `radial` bonds away from `a1`
  adj_list = set(canon_adj_list[a1])
  # atoms less than `radial` bonds away
  all_list = set([a1])
  while radial < max_distance:
    distance[list(adj_list), radial] = 1
    all_list.update(adj_list)
    # find atoms `radial`+1 bonds away
    next_adj = set()
    for adj in adj_list:
      next_adj.update(canon_adj_list[adj])
    adj_list = next_adj - all_list
    radial = radial + 1
  return distance

if __name__ == '__main__':
    # Get the sdf data ready, "all.sdf", "all.sdf.csv" two files in the same folder;
    dataset_file = 'ac19/10.sdf'

    # Get to know the tasks
    qm8_tasks = [
        "property_0", "property_1", "property_2", "property_3", "property_4",
        "property_5", "property_6", "property_7", "property_8", "property_9",
        "property_10", "property_11"
    ]

    if featurizer == 'MP':
        featurizer = dc.feat.WeaveFeaturizer(
            graph_distance=False, explicit_H=True)

    # Utilize deepchem's tool to load SDF file
    loader = dc.data.SDFLoader(
        tasks=qm8_tasks,
        smiles_field="smiles",
        mol_field="mol",
        featurizer=featurizer)

    dataset = loader.featurize(dataset_file)

    print('Featurization finished!')


    print("Pair features: ", dataset.get_pair_features())

    print("Node featuers: ", dataset.get_atom_features())

    print("There are ", dataset.get_num_atoms(), " atoms.")

    print("There are ", dataset.get_num_features(), "features. ")
