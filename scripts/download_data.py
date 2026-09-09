"""Download the MaleCNS v1.0 flat-connectome files into data/."""
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {
    "body-annotations.feather": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters.feather": "body-neurotransmitters-male-cns-v1.0.feather",
    "edges-traced.feather": "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
}


def main(data_dir: Path = Path("data")) -> None:
    data_dir.mkdir(exist_ok=True)
    for local, remote in FILES.items():
        dest = data_dir / local
        if dest.exists():
            print(f"skip {local}")
            continue
        print(f"fetch {remote}")
        urllib.request.urlretrieve(BASE + remote, dest)


if __name__ == "__main__":
    main()
