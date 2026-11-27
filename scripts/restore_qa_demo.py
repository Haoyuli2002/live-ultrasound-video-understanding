#!/usr/bin/env python3
"""Verify, join and extract the seven parts of the frozen ultrasound QA demo."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def restore(base):
    manifest = json.loads((base / 'download_manifest.json').read_text())
    combined = hashlib.sha256()
    for part in manifest['parts']:
        path = base / part['name']
        if not path.is_file():
            raise FileNotFoundError('Missing download: ' + path.name)
        if path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
            raise ValueError('Incomplete or damaged download: ' + path.name)
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                combined.update(chunk)
        print('Verified', path.name, flush=True)
    if combined.hexdigest() != manifest['archive_sha256']:
        raise ValueError('Combined archive checksum mismatch')
    target = base / manifest['archive_name']
    if target.exists():
        if digest(target) != manifest['archive_sha256']:
            raise FileExistsError('A different archive already exists: ' + str(target))
    else:
        # Exclusive creation preserves any existing user file.
        with target.open('xb') as output:
            for part in manifest['parts']:
                with (base / part['name']).open('rb') as stream:
                    shutil.copyfileobj(stream, output, 4 * 1024 * 1024)
        if digest(target) != manifest['archive_sha256']:
            raise ValueError('Written archive checksum mismatch; keep the parts and retry in a new folder')
    folder = base / 'Ultrasound-QA-10-Examples'
    if folder.exists():
        print('Archive verified; existing extracted folder preserved:', folder)
        return target
    with zipfile.ZipFile(target) as archive:
        for member in archive.infolist():
            path = Path(member.filename)
            if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != folder.name:
                raise ValueError('Unexpected archive path: ' + member.filename)
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError('ZIP checksum failed: ' + corrupt)
        with tempfile.TemporaryDirectory(prefix='qa-demo-extract-', dir=base) as temporary:
            archive.extractall(temporary)
            Path(temporary, folder.name).rename(folder)
    print('\nReady:', folder)
    print('Run python3 serve.py from that folder, or select its media/*.mp4 in the lightweight demo.')
    return target


if __name__ == '__main__':
    restore(Path(__file__).resolve().parent)
