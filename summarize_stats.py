import json
import glob
import os
import pandas as pd


def summarize(fake_real_summarize='fake', work_dir='/media/linuxu/150b6498-cf67-4ecf-a12d-c66348446ad1/'):
    table_columns = ['file_name', 'detection_model', 'total_fake_frames', 'total_real_frames', 'percent_fake_frame',
                     'percent_real_frame', 'detector_classification', 'real_classification']
    summery_table = pd.DataFrame(columns=table_columns)
    fake_real_path = 'detection_videos/'
    models_names = ['mesoNet', 'xception']
    row = dict.fromkeys(table_columns, None)
    for model_name in models_names:
        json_files = glob.glob(f'{work_dir + fake_real_path + model_name}/*.json')
        for json_path in json_files:
            with open(json_path) as f:
                file_name = os.path.basename(json_path)
                data = json.load(f)
                row['file_name'] = file_name
                row['detection_model'] = model_name
                row['total_fake_frames'] = data['total_fake_frames']
                row['total_real_frames'] = data['total_real_frames']
                row['percent_fake_frame'] = data['percent_fake_frames']
                row['percent_real_frame'] = 1 - data['percent_fake_frames']
                row['detector_classification'] = 'fake' if data['percent_fake_frames'] > 0.5 else 'real'
                row['real_classification'] = 'fake' if 'fake' in fake_real_path else 'real'
                summery_table = pd.concat([summery_table, pd.DataFrame([row])], ignore_index=True)
    return summery_table


def calc_stats(df, model_name=None):
    assert model_name is not None, "must input model name"
    model_rows = df.loc[df.detection_model == model_name]
    mean_real_frames_percent = model_rows.percent_real_frame.mean()
    mean_fake_frames_percent = model_rows.percent_fake_frame.mean()
    mean_total_frame = (model_rows.total_real_frames + model_rows.total_fake_frames).mean()
    fake_count = (model_rows.detector_classification == 'fake').sum()
    real_count = (model_rows.detector_classification == 'real').sum()
    fake_precision = fake_count/model_rows.shape[0]
    real_precision = real_count/model_rows.shape[0]
    total_fake_vids = (model_rows.real_classification == 'fake').sum()
    total_real_vids = (model_rows.real_classification == 'real').sum()
    total_dict = {"mean_real_frames_percent": mean_real_frames_percent,
                  "mean_fake_frames_percent": mean_fake_frames_percent,
                  "mean_total_frame": mean_total_frame,
                  "detector_fake_count": fake_count,
                  "detector_real_count": real_count,
                  "detector_fake_precision": fake_precision,
                  "detector_real_precision": real_precision,
                  "total_fake_videos": total_fake_vids,
                  "total_real_videos": total_real_vids
                  }
    return pd.Series(data=total_dict)


if __name__ == '__main__':
    work_dir = '/media/linuxu/150b6498-cf67-4ecf-a12d-c66348446ad1/C23/sunset_original_videos'
    real_dataset_summ = summarize(fake_real_summarize='real', work_dir=work_dir)
    fake_dataset_summ = summarize(fake_real_summarize='fake', work_dir=work_dir)

    print("============Real Dataset stats============")
    xception_real_stats = calc_stats(real_dataset_summ, 'xception')
    meso_real_stats = calc_stats(real_dataset_summ, 'mesoNet')
    print("=====Xception=====")
    print(xception_real_stats)
    print("\n")
    print("=====MesoNet=====")
    print(meso_real_stats)
    print("\n")
    print("============Fake Dataset stats============")
    xception_fake_stats = calc_stats(fake_dataset_summ, 'xception')
    meso_fake_stats = calc_stats(fake_dataset_summ, 'mesoNet')
    print("=====Xception=====")
    print(xception_fake_stats)
    print("\n")
    print("=====MesoNet=====")
    print(meso_fake_stats)
    pass
