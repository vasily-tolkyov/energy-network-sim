import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
const base=path.dirname(fileURLToPath(import.meta.url));
const read=p=>fs.readFileSync(path.join(base,p),'utf8').replace(/^\uFEFF/,'');
const check=JSON.parse(read('evidence-check.json')),integrity=JSON.parse(read('source-integrity.json'));
assert.deepEqual(check.pending,[]);assert.equal(check.reportHasPlaceholder,false);assert.equal(check.checkedQueries,1972);
assert.ok(integrity.trackedFilesUnchanged);assert.equal(integrity.commit,'e732f2b4ef46e0ac482480095a31cb5aa23ec6be');
assert.ok(!/<!--\s*[A-Z_]+\s*-->/.test(read('REVIEW-REPORT.zh-CN.md')));
const files=[];function walk(p){for(const e of fs.readdirSync(p,{withFileTypes:true})){const q=path.join(p,e.name);if(e.isDirectory())walk(q);else if(!['manifest-sha256.json','REVIEW-STATE.md'].includes(e.name))files.push(q);}}
walk(base);
const manifest={commit:integrity.commit,created:new Date().toISOString(),files:files.sort().map(p=>({path:path.relative(base,p).replaceAll('\\','/'),bytes:fs.statSync(p).size,sha256:crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex')}))};
fs.writeFileSync(path.join(base,'manifest-sha256.json'),JSON.stringify(manifest,null,2));console.log(JSON.stringify({files:manifest.files.length}));
