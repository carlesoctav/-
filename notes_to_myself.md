
thinks about 
devices, accelerator params inside TrainerUtil Class



multi device training setup:
1. load the same dataset fr ech process 
2. gave a global named sharding over "data"  axis


multi device training setup:
1. load the same dataset fr ech process 
2. use distributed sampler so ech process will receive different batch from the data.

