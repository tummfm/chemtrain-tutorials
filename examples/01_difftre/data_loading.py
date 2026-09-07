"""Reference data loader for the DiffTRe water tutorial."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.request import urlretrieve

import jax.numpy as jnp
import numpy as np

from chemtrain.data import preprocessing


POSITIONS_URL = (
    "https://drive.usercontent.google.com/download?id=1wVJ3cEakl5IngvuZycp0wEOsn7EQyBD8"
    "&export=download&authuser=0&confirm=t&uuid=916ca889-af2d-458f-8e97-5b1c23ea39ba"
    "&at=ALBwUgm15BEzvzbLeMs6UUzr45Lb:1777900857190"
)
FORCES_URL = (
    "https://drive.usercontent.google.com/download?id=1EgSrBa5e2-dlgG0X8fIDATbR9q3xSFwv"
    "&export=download&authuser=0&confirm=t&uuid=994f428e-a771-427f-b769-fce10ec66dd1"
    "&at=ALBwUglSh_WZNFUOOdddmeZC3JTT:1777900862535"
)
RDF_URL = "https://raw.githubusercontent.com/tummfm/difftre/92c0790b89f0d570ed9f79663e6c06580f598345/data/experimental/O_O_RDF.csv"
ADF_URL = "https://raw.githubusercontent.com/tummfm/difftre/92c0790b89f0d570ed9f79663e6c06580f598345/data/experimental/O_O_O_ADF.csv"

_CHECKSUMS = {
    "positions.npy": "873c7ee08bd6238531ec10dbc41e2f9651d128cdc6c61dd7dfa29f22e931bdbc",
    "forces.npy": "b724eb35cc1890f5866decb0ac9b04a3f4a624bf9a323ceb89d2e99314a43687",
    "O_O_RDF.csv": "b9701ad4dd35b47985a5f4e4cd01b8351ae384a99282ad8f91dfe9e5be555192",
    "O_O_O_ADF.csv": "a737a25056524f04b461d899726289bfe77a4ca1308dad6151a053e0d8651613",
}


@dataclass(frozen=True)
class WaterReferenceData:
    box: jnp.ndarray
    positions: jnp.ndarray
    forces: jnp.ndarray
    rdf_distances: np.ndarray
    rdf: np.ndarray
    adf_angles: np.ndarray
    adf: np.ndarray

    @property
    def dataset(self):
        return {"R": self.positions, "F": self.forces}


def _download_if_missing(url: str, path: Path) -> None:
    if not path.exists():
        urlretrieve(url, path)
    assert sha256(path.read_bytes()).hexdigest() == _CHECKSUMS[path.name]


def load_coarse_grained_water(
    data_dir: Path = Path("data"), subsampling: int = 10
) -> WaterReferenceData:
    """Load CG-water positions, forces, and experimental RDF/ADF targets."""
    data_dir.mkdir(exist_ok=True)
    paths = {
        "positions.npy": data_dir / "positions.npy",
        "forces.npy": data_dir / "forces.npy",
        "O_O_RDF.csv": data_dir / "O_O_RDF.csv",
        "O_O_O_ADF.csv": data_dir / "O_O_O_ADF.csv",
    }
    for name, url in {
        "positions.npy": POSITIONS_URL,
        "forces.npy": FORCES_URL,
        "O_O_RDF.csv": RDF_URL,
        "O_O_O_ADF.csv": ADF_URL,
    }.items():
        _download_if_missing(url, paths[name])

    box = jnp.full(3, 3.12867066)
    positions = preprocessing.scale_dataset_fractional(
        preprocessing.get_dataset(paths["positions.npy"], subsampling=subsampling), box
    )
    forces = preprocessing.get_dataset(paths["forces.npy"], subsampling=subsampling)
    rdf_distances, rdf = np.loadtxt(paths["O_O_RDF.csv"], unpack=True)
    adf_angles, adf = np.loadtxt(paths["O_O_O_ADF.csv"], unpack=True)

    return WaterReferenceData(
        box=box,
        positions=positions,
        forces=forces,
        rdf_distances=rdf_distances,
        rdf=rdf,
        adf_angles=adf_angles,
        adf=adf,
    )
