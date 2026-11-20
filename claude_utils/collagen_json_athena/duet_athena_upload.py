"""Upload the DUET Aim3 H&E slides (<slide>/B.svs) and their cortex/medulla mask predictions to Athena (DSA).

  python duet_athena_upload.py --collection 6b005d3fec7333818fc7b969 \
      --images /orange/.../aim3/images --masks /blue/.../cortex_medulla/all/inference_jsons/svc_2_rbf_1.0

Auth: the API key is read from the ATHENA_API_KEY environment variable (never from the command line).
Each slide becomes item "<slide>_B.svs" in folder --folder of the collection, matching the sample names of
the mask JSONs (<slide>_B_mask_prediction.json). Rerunnable: slides already uploaded and annotations with
the same name already on the item are skipped.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import girder_client

API = "https://athena.rc.ufl.edu/api/v1/"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collection", required=True)
    p.add_argument("--folder", default="DUET_Aim3_T0")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--mask-suffix", default="_mask_prediction.json",
                   help="annotation file = <masks>/<slide>_B<suffix> (collagen: _collagen_mask.json)")
    p.add_argument("--only", nargs="*", help="slide folder names (default: all)")
    a = p.parse_args()

    key = os.environ.get("ATHENA_API_KEY")
    if not key:
        sys.exit("ATHENA_API_KEY is not set")
    gc = girder_client.GirderClient(apiUrl=API)
    gc.authenticate(apiKey=key)
    me = gc.get("user/me")
    print(f"authenticated as {me['login']}", flush=True)

    folder = gc.createFolder(a.collection, a.folder, parentType="collection", reuseExisting=True)
    items = {it["name"]: it for it in gc.listItem(folder["_id"])}
    slides = a.only or sorted(d.name for d in a.images.iterdir() if (d / "B.svs").is_file())
    n_up = n_ann = 0
    for s in slides:
        name = f"{s}_B.svs"
        src = a.images / s / "B.svs"
        if name not in items:
            print(f"upload {name} ({src.stat().st_size / 1e6:.0f} MB)", flush=True)
            gc.uploadFileToFolder(folder["_id"], str(src), filename=name)
            items.update({it["name"]: it for it in gc.listItem(folder["_id"], name=name)})
            n_up += 1
        item = items[name]
        if "largeImage" not in gc.getItem(item["_id"]):
            try:
                gc.post(f"item/{item['_id']}/tiles")
            except girder_client.HttpError as e:
                print(f"  note: could not start large-image conversion for {name}: {e}", flush=True)

        mask = a.masks / f"{s}_B{a.mask_suffix}"
        if not mask.is_file():
            print(f"  no mask yet for {s}", flush=True)
            continue
        have = {an["annotation"]["name"] for an in gc.get("annotation", parameters={"itemId": item["_id"],
                                                                                   "limit": 0})}
        for doc in json.loads(mask.read_text()):
            if doc["name"] in have:
                continue
            gc.post("annotation", parameters={"itemId": item["_id"]}, json=doc)
            n_ann += 1
            print(f"  annotation '{doc['name']}' ({len(doc['elements'])} polygons) -> {name}", flush=True)
    print(f"done: {n_up} slides uploaded, {n_ann} annotations added, folder {a.folder}", flush=True)


if __name__ == "__main__":
    main()
