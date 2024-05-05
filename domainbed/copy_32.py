import os
impoot json
input_dir = "./results_vits_3600"
count = 0
if not os.path.exists("./results_vits_3600_32"):
    os.makedirs("./results_vits_3600_32")
for direct in os.listdir(input_dir):
    # read the fist row of results.jsonl
    with open(os.path.join(input_dir, direct, "results.jsonl")) as f:
        first_line = f.readline()
        first_line = json.loads(first_line)
    if first_line['hparams']['batch_size'] == 32:
    # copy the direct to the new directory
        os.system(f"cp -r {os.path.join(input_dir, direct)} ./results_vits_3600_32")
        count += 1
