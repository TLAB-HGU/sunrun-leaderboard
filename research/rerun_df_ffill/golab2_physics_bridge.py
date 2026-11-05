"""Send the verified local physics prerequisite to the reserved golab2 queue."""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tarfile
import time

import launch
from golab2_transport import remote

ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = '/home/yehwan/sunrun-golab2-20261103-2300'


def main():
    receipt = ROOT / 'golab2-physics-transfer.json'
    if receipt.exists():
        return
    while True:
        try:
            proof = launch.physics_dependency()
        except (OSError, ValueError, KeyError) as error:
            print(json.dumps({'waiting_for_physics': str(error)}), flush=True)
            time.sleep(60)
            continue
        package = io.BytesIO()
        with tarfile.open(fileobj=package, mode='w') as archive:
            for name in ('input/huxt_features.parquet', 'input/huxt_features.metadata.json',
                         'gong_cache/exit.code'):
                archive.add(ROOT / name, arcname=name)
        try:
            result = remote(
                'import tarfile,sys;tarfile.open(fileobj=sys.stdin.buffer,mode="r|")'
                '.extractall(' + repr(REMOTE_ROOT) + ',filter="data");print("physics transferred")',
                input=package.getvalue(), timeout=180)
        except Exception as error:
            print(json.dumps({'transfer_retry': str(error)}), flush=True)
            time.sleep(60)
            continue
        receipt.write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
                                       'host': 'golab2', 'local_validation': proof,
                                       'remote_receipt': result.stdout.decode()}, indent=2) + '\n')
        print('Verified physics transferred', flush=True)
        return


if __name__ == '__main__':
    main()
