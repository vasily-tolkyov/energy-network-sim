import test from 'node:test';
import assert from 'node:assert/strict';
import { EnergyNetwork } from '../repo/dist/src/network.js';
import { mulberry32 } from '../repo/dist/src/prng.js';
const make = (n=3) => new EnergyNetwork({neuronCount:n,activationEnergy:1,maintenanceEnergy:0,maxWeight:10,maxDirectedWeight:10});
const pattern = (net,ids=net.activeNeurons()) => { const p=new Uint8Array(net.neuronCount); for(const i of ids)p[i]=1; return p; };
const residual = (net,clamps) => { const s=pattern(net); return [...s.keys()].filter(i=>!clamps.includes(i)).map(i=>({i,delta:(s[i]===0?1:-1)*(net.threshold-net.symmetricField(i,s))})).filter(x=>x.delta < -1e-4); };
const oscillator = () => {const n=make();n.strengthen(0,1,2);n.strengthen(1,2,2);n.strengthenDirectedInhibitory(2,1,5);return n;};

test('A01 symmetric W/Gamma: independent energy and fixed-point oracle (40 seeds)',()=>{
  for(let seed=1;seed<=40;seed++){
    const n=make(8),rand=mulberry32(seed);
    for(let i=0;i<8;i++)for(let j=i+1;j<8;j++){n.strengthen(i,j,rand()*2);n.strengthenInhibitory(i,j,rand());}
    const r=n.settle([0]);assert.equal(r.energy,n.energy(pattern(n)));
    assert.ok(Math.abs(r.trace.energies.at(-1)-r.energy)<1e-9);
    assert.equal(residual(n,[0]).length,0);
    for(let k=1;k<r.trace.energies.length;k++)assert.ok(r.trace.energies[k] < r.trace.energies[k-1]);
  }
});

test('A02 directed inhibition: trace must agree with reported physical energy',()=>{
  const n=make(2);n.strengthen(0,1,2);n.strengthenDirectedInhibitory(0,1,0.5);
  const r=n.settle([0]);console.log('A02 evidence',JSON.stringify(r));
  assert.ok(Math.abs(r.trace.energies.at(-1)-r.energy)<1e-9,'trace uses nonconservative work as state energy');
});

test('A03 DI oscillator: returned result must be stable or explicitly nonconverged',()=>{
  const n=oscillator(),r=n.settle([0]),unstable=residual(n,[0]);
  console.log('A03 evidence',JSON.stringify({active:r.activeNeurons,energy:r.energy,traceEnd:r.trace.energies.at(-1),flips:r.trace.flipCount,unstable,keys:Object.keys(r)}));
  assert.ok(!unstable.length || r.converged===false,'unstable result has no convergence status');
});

test('A04 bounded quench must restore a true minimum-energy visited state',()=>{
  const n=oscillator();const r=n.settleAnnealed([0],[],{levels:1,sweepsPerLevel:1,quenchMaxFlips:3});
  const visitedMinimum=n.energy(pattern(n,[0,1,2]));
  console.log('A04 evidence',JSON.stringify({result:r,visitedMinimum,unstable:residual(n,[0])}));
  assert.ok(r.energy <= visitedMinimum+1e-9,'fallback restores state energy 1 although -1 was visited');
});

test('A05 weight API must reject out-of-range neuron IDs without destroying symmetry',()=>{
  const n=make();let threw=false;try{n.strengthen(0,3,0.75)}catch{threw=true}
  console.log('A05 evidence',JSON.stringify({threw,W10:n.getWeight(1,0),W01:n.getWeight(0,1)}));
  assert.ok(threw,'out-of-range ID accepted');assert.equal(n.getWeight(1,0),n.getWeight(0,1));
});

test('A06 fractional/NaN clamped inputs must be rejected rather than silently disappear',()=>{
  for(const i of [0.5,NaN])assert.throws(()=>make().settle([i]),`invalid index ${i} accepted`);
});

test('A07 nonfinite learning coefficients must be rejected',()=>{
  assert.throws(()=>new EnergyNetwork({neuronCount:3,activationEnergy:1,maintenanceEnergy:0,learningRate:NaN}));
});

test('A08 nonfinite edge values must be rejected, not poison dynamics',()=>{
  const n=make();assert.throws(()=>n.setWeight(0,1,NaN));
});

test('A09 fixed seed annealing determinism and input clamping (20 seeds)',()=>{
  for(let seed=1;seed<=20;seed++){
    const n=make(6);for(let i=0;i<6;i++)for(let j=i+1;j<6;j++)n.strengthen(i,j,0.7);
    const opts={seed,levels:5,sweepsPerLevel:4,quenchCandidatesOnly:true};
    const wells=[{memberNeuronIds:[0,1,2,3,4,5]}];
    const a=n.settleAnnealed([0],wells,opts),b=n.settleAnnealed([0],wells,opts);
    assert.deepEqual(a,b);assert.ok(a.activeNeurons.includes(0));assert.equal(residual(n,[0]).length,0);
  }
});

test('A10 candidate-restricted quench correctly prevents out-of-scope recruitment',()=>{
  const n=make(2);n.strengthen(0,1,2);const r=n.settleAnnealed([0],[],{levels:1,sweepsPerLevel:1,quenchCandidatesOnly:true});
  assert.deepEqual(r.activeNeurons,[0]);
});

test('A11 cold symmetric settling is local, not a global-energy optimizer (boundary control)',()=>{
  const n=make(4);for(let i=0;i<4;i++)for(let j=i+1;j<4;j++)n.strengthen(i,j,1);
  const r=n.settle([]);assert.equal(r.energy,0);assert.equal(n.energy(pattern(n,[0,1,2,3])),-2);
});

test('A12 quench hard flip budget must not be exceeded',()=>{
  const n=oscillator();const r=n.settleAnnealed([0],[],{levels:1,sweepsPerLevel:1,quenchMaxFlips:1});
  console.log('A12 evidence',JSON.stringify({requestedLimit:1,reportedFlips:r.trace.flipCount,trace:r.trace.energies}));
  assert.ok(r.trace.flipCount<=1,'limit only checked after a complete sweep, fallback also counted as a flip');
});
