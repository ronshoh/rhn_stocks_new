import os
import sys
import boto3
import argparse
import time
import logging
logging.basicConfig(format='%(asctime)-15s | %(levelname)-8s | %(message)s')

SCORE_FILENAME = "final_predictions"


class S3(object):
    def __init__(self, region=None, aws_access_key_id=None, aws_secret_access_key=None):
        # Get the service resource
        self.s3 = boto3.resource('s3', region_name=region, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
        self.log = logging.getLogger('s3')
        self.log.setLevel(logging.INFO)

    def set_log_level(self, lvl):
        self.log.setLevel(lvl)

    ## @brief      Get S3 objects list.
    def objects(self, bucket, prefix):
        for obj in self.s3.Bucket(bucket).objects.filter(Prefix=prefix):
            if obj.storage_class == 'STANDARD':
                yield obj


    ## @brief      Put S3 object file.
    ##
    ## @param[in]  bucket is the S3 bucket name.
    ## @param[in]  key is the S3 object key name.
    ## @param[in]  filename is the name of the file to upload.

    def put_file(self, bucket, key, filename):
        try:
            self.log.info('uploading data file {filename} to s3://{bucket}/{key}.'.format(filename=filename, bucket=bucket, key=key))
            with open(filename, 'rb') as f:
                self.s3.Bucket(bucket).put_object(Body=f, Key=key)
            self.log.info('upload completed.\n')
            return 0
        except Exception:
            self.log.exception('got exception while trying to upload {filename} to S3.'.format(filename=filename))
        return -1

    def get_file(self, bucket, key, down_path):
        wall_time = 2
        while True:
            try:
                self.log.info('downloading data file from s3://{bucket}/{key} to {down_path}'.format(down_path=down_path, bucket=bucket, key=key))
                self.s3.Bucket(bucket).download_file(Key=key, Filename=down_path)
                self.log.info('download completed.\n')
                return 0
            except Exception:
                wall_time -= 1
                if wall_time > 0:
                    time.sleep(1)
                    continue
                self.log.exception('got exception while trying to download {filename} from S3.'.format(filename=down_path))
                return -1


def get_simulation_dir(root):
    for (folder, subs, files) in os.walk(root):
        for filename in files:
            scores_file = [f for f in files if f.startswith(SCORE_FILENAME) and f.endswith("txt")]
            if filename == SCORE_FILENAME + ".mat" and len(scores_file) == 0:
                sim_name = folder.split("\\")[-1]
                file_path = os.path.join(folder, filename)
                yield(sim_name, file_path)


def write2summary(summary, down_dir, scores_path):
    f = open(summary, "r")
    lines = f.readlines()
    f.close()

    f = open(scores_path, "r")
    s_lines = f.readlines()
    f.close()
    scores = [s_lines[0][:-2], s_lines[1][:-2]]

    new_lines = []
    for i, line in enumerate(lines):
        if i == 0:
            new_lines.append(line)
            continue
        words = line.split(",")
        sim_name = "".join(down_dir.split("\\")[-1])
        if words[0].strip() == sim_name.strip():
            words[-1] = words[-1][:-2]
            words.extend(scores)
            new_lines.append(",".join(words) + "\n")
        else:
            new_lines.append(line)

    f = open("./results/summary.csv", "w")
    for line in new_lines:
        f.write(line)
    f.close()

    return


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dir", type=str, default=None, help="results' dir")
    ap.add_argument("--s3-path", type=str, default=None, help="S3 Path. (haim folder is excluded)")
    ap.add_argument("--summary", type=str, default=None, help="summary file Path.")
    ap.add_argument("--max_files", type=int, default=None, help="max files to upload in a single run")
    ap.add_argument("--wait_time", type=int, default=600, help="max files to upload in a single run")

    args = ap.parse_args()

    s3 = S3('us-east-1', 'AKIAIYY3BXHCOP6ZVPNA', 'SLcm0GNVY9b/NXSgOA+1BJplDKAEaadp/epS74j3')


    s3.log.info("deleting previous results...\n\n")
    for obj in s3.objects('btx-job-request', 'haim/{s3_path}'.format(s3_path=args.s3_path).replace('//', '/')):
        obj.delete()

    sim_list = []
    s3.log.info("uploading files...\n\n")
    for i, (sim_name, file_path) in enumerate(get_simulation_dir(args.dir)):
        sim_list.append((i, sim_name, file_path))
        s3_path = args.s3_path + '{:04d}'.format(i) + "/"
        s3.put_file('btx-job-request',
                    'haim/{s3_path}/{filename}'.format(s3_path=s3_path, filename=os.path.split(file_path)[-1]).replace('//','/'),
                    file_path)
        if args.max_files:
            if i == args.max_files - 1:
                break

    s3.log.info("waiting for results %d seconds...\n\n" % (args.wait_time))
    time.sleep(args.wait_time)

    s3.log.info("downloading files...\n\n")
    success_arr = [0] * len(sim_list)
    for (i, sim_name, file_path) in sim_list:
        print("current dir path:  haim/{s3_path}/{i}/".format(s3_path=args.s3_path, i='{:04d}'.format(i)).replace('//', '/'))
        for obj in s3.objects('btx-job-request', 'haim/{s3_path}/{i}/'.format(s3_path=args.s3_path, i='{:04d}'.format(i)).replace('//', '/')):
            file_name = obj.key.split("/")[-1]
            if  file_name.startswith(SCORE_FILENAME) and file_name.endswith("txt"):
                s3_path = args.s3_path + '{:04d}'.format(i)
                down_dir = "\\".join(file_path.split("\\")[:-1])
                down_path = down_dir + "\\" + SCORE_FILENAME + ".txt"
                success_arr[i] =  s3.get_file('btx-job-request', 'haim/{s3_path}/{filename}'.format(s3_path=s3_path,
                                                                                                    filename=file_name),
                                                                                                    down_path=down_path)
                if args.summary and success_arr[i] == 0:
                    write2summary(args.summary, down_dir, down_path)

                if args.max_files:
                    if i == args.max_files - 1:
                        break



    return 0

## -- ##

if __name__ == "__main__":
    sys.exit(main())
