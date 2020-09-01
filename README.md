# mrpc-classification

instructions to reproduce 90.2 acc and 85 F1 score

  ```
	sudo docker pull iyernikhil007/test-docker
	docker run --name test-docker -it -v ${YOUR_CODE_DIR}:/data  iyernikhil007/test-docker bash
  ```
  
To train the model:

```
CUDA_VISIBLE_DEVICES=0 python run_glue.py     --model_type bert     --model_name_or_path bert-base-uncased     --task_name MRPC     --do_train     --do_eval     	  --do_lower_case     --data_dir MRPC     --max_seq_length 128     --per_gpu_eval_batch_size=8       --per_gpu_train_batch_size=8       		    	     --learning_rate 2e-5     --num_train_epochs 2.0     --output_dir ${CKPT_DIR}  --overwrite_output_dir
  	  ```
	 
To test the model:

```
	export FLASK_APP=server.py
        flask run
	curl --header "Content-Type: application/json" \
  	--request POST \
  	--data '{"input1":${SENTENCE1}, "input2":${SENTENCE2}' \
  	http://localhost:5000
	```


for changing the args for testing, please look at test_args.json
