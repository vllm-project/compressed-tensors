def apply_module_parallel(modules, apply_fn, weight_fn):
	_, _, assigned_rank = greedy_bin_packing(modules, dist.get_world_size(), weight_fn)
	
	with disable_onloading():
		for module in modules:
		  if assigned_rank[module] != dist.get_rank():
		    to_meta(module)  # 1. remove non-processing rank pointers
		    apply_fn(module)  # 2. compress on meta (prepare step 4)
	
	with as_single_threaded():
		for module in modules:
		  if assigned_rank[module] == dist.get_rank():
		    apply_fn(module)  # 3. compress without triggering sync
	
	for module in modules:
		with set_main_process(assigned_rank[module]):
		  with disable_onloading():
		    state_dict = get_direct_state_dict(module)
		  replace_direct_state_dict(module, state_dict)  # 4. broadcast source offload across ranks