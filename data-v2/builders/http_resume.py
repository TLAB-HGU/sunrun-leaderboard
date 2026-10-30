"""Resume interrupted public downloads, verifying Range and final body length."""
import hashlib, json, re, time, http.client
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError

class DownloadCapExceeded(ValueError):
    """Declared or received body exceeds the caller's explicit byte budget."""

def download(url,path,attempts=12,max_bytes=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);meta_path=path.with_suffix(path.suffix+'.http.json')
    meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for attempt in range(attempts):
        offset=path.stat().st_size if path.exists() else 0
        if max_bytes is not None and offset>max_bytes:raise DownloadCapExceeded('cached body exceeds byte budget')
        # A partial body without its validator is not safe to append.
        if offset and not (meta.get('etag') or meta.get('http_last_modified')):
            path.unlink();offset=0
        if offset and offset==meta.get('content_length'): break
        headers={'User-Agent':'sunrun-data-v2/1.0','Accept-Encoding':'identity'}
        if offset:
            headers.update(Range=f'bytes={offset}-',**{'If-Range':meta.get('etag') or meta['http_last_modified']})
        try:
            started=time.monotonic()
            with urlopen(Request(url,headers=headers),timeout=45) as response:
                if offset and response.status==200:offset=0
                if response.status==206:
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                    if not match or int(match[1])!=offset:raise ValueError('invalid resumed Content-Range')
                    total=int(match[3])
                elif response.status==200:
                    length=response.headers.get('Content-Length')
                    total=int(response.headers['Content-Length']) if max_bytes is None or length is not None else None
                else:raise ValueError(f'HTTP {response.status}')
                if max_bytes is not None and total is not None and total>max_bytes:raise DownloadCapExceeded('Content-Length exceeds byte budget')
                meta={'url':url,'content_length':total,'http_last_modified':response.headers.get('Last-Modified'),'etag':response.headers.get('ETag'),'retrieved_at':datetime.now(timezone.utc).isoformat()}
                meta_path.write_text(json.dumps(meta))
                with path.open('ab' if offset else 'wb') as stream:
                    while block:=response.read(64*1024):
                        if max_bytes is not None and stream.tell()+len(block)>max_bytes:raise DownloadCapExceeded('stream exceeds byte budget')
                        stream.write(block)
                        if time.monotonic()-started>300:raise TimeoutError('five-minute response budget; resume next request')
                if total is not None and path.stat().st_size!=total:raise IOError('truncated HTTP body')
                break
        except DownloadCapExceeded:
            raise
        except HTTPError as exc:
            if exc.code in (403,404):raise
            if attempt+1==attempts:raise
            time.sleep(2)
        except (OSError,ValueError,http.client.HTTPException) as exc:
            print('download retry',url,attempt+1,str(exc),flush=True)
            if attempt+1==attempts:raise
            time.sleep(2)
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return dict(meta,status=200,bytes=path.stat().st_size,sha256=digest.hexdigest())
