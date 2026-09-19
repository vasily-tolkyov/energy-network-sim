import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
// These tests assert saved measurements; actual world executions are new-worlds.mjs.
for(let k=1;k<=5;k++){const id=`W${String(k).padStart(2,'0')}`,r=JSON.parse(readFileSync(new URL(`../results/${id}-new-world.json`,import.meta.url),'utf8')).summary;test(`${id}: independent world has >=90% recall and all true factors`,()=>{assert.ok(r.passesAccuracy,JSON.stringify(r));assert.ok(r.passesFactors,JSON.stringify(r));});}
