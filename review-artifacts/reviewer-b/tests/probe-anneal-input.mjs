import {EnergyNetwork} from '../repo/dist/src/network.js';
for(const seed of [1,2,3]){
 const n=new EnergyNetwork({neuronCount:4,activationEnergy:1,maintenanceEnergy:.5,maxWeight:10});
 // Candidate 1.5 is not a valid neuron. It aliases typed-array row index offsets during field reads.
 n.strengthen(1,2,4);
 for(const opts of [{extraCandidates:[1.5],quenchCandidatesOnly:true,levels:1,sweepsPerLevel:1,seed}, {levels:NaN,seed}, {initialTemperature:NaN,levels:1,sweepsPerLevel:1,seed}]){
 try {const r=n.settleAnnealed([0],[],opts);console.log(JSON.stringify({opts,r}));}catch(e){console.log('THREW',String(e));}
 }
}
