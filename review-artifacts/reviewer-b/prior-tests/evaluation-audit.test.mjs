import test from 'node:test';
import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {labProbes} from '../repo/dist/src/topics/lab-world.js';

test('E01 measurement: constant-on baseline is perfect on the named heldout coarse stratum',()=>{
 const probes=labProbes(new Set()),heldout=probes.filter(p=>p.kind==='heldout');
 const baseline=heldout.filter(p=>p.truth.lit===1).length/heldout.length;
 const record={probes:probes.length,heldout:heldout.length,gateOff:probes.filter(p=>p.kind==='gate-off').length,heldoutClasses:[...new Set(heldout.map(p=>p.truth.lit))],constantOnHeldoutCoarse:baseline,constantOffAllCoarse:probes.filter(p=>p.truth.lit===0).length/probes.length};
 writeFileSync(new URL('../results/E01-metric-baselines.json',import.meta.url),JSON.stringify(record,null,2));
 console.log('E01',JSON.stringify(record));
 assert.equal(baseline,1);assert.deepEqual(record.heldoutClasses,[1]);
 // Descriptive audit result. PASS confirms the metric degeneracy, not model generalization.
});

test('E02 identifiability boundary: provided continuous switch grid cannot identify threshold 0.5 exactly',()=>{
 // LAB_CONT_GRID.switchPos from the pinned source; not a claim of executing ContinuousExplorer.
 const grid=[0.1,0.5,0.9];
 const observations=t=>grid.map(x=>Number(x>=t));
 const a=observations(0.2),b=observations(0.5);
 console.log('E02',JSON.stringify({grid,thresholds:[0.2,0.5],observations:[a,b]}));
 assert.deepEqual(a,b); // A concept center at supplied x=0.5 is not an identified causal boundary.
});
