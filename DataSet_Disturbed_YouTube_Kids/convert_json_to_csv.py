import json
import csv
import os
import re

def parse_iso_duration(duration_str):
    if not duration_str or not isinstance(duration_str, str):
        return None
    pattern = r'P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?'
    match = re.match(pattern, duration_str)
    if not match:
        return None
    parts = match.groupdict()
    days = int(parts['days'] or 0)
    hours = int(parts['hours'] or 0)
    minutes = int(parts['minutes'] or 0)
    seconds = int(parts['seconds'] or 0)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds

def safe_int(val):
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None

def extract_video_row(obj, source_name):
    snippet = obj.get('snippet', {}) or {}
    stats = obj.get('statistics', {}) or {}
    content = obj.get('contentDetails', {}) or {}

    tags_list = snippet.get('tags')
    tags_str = '|'.join(tags_list) if isinstance(tags_list, list) else ''

    duration_iso = content.get('duration', '')
    duration_sec = parse_iso_duration(duration_iso)

    is_gt = 1 if (source_name == 'groundtruth' or obj.get('isGroundTruthVideo') == 1 or obj.get('classification_label')) else 0

    return {
        'video_id': obj.get('id', ''),
        'title': snippet.get('title', ''),
        'description': snippet.get('description', ''),
        'channel_id': snippet.get('channelId', ''),
        'channel_title': snippet.get('channelTitle', ''),
        'published_at': snippet.get('publishedAt', ''),
        'category_id': snippet.get('categoryId', ''),
        'tags': tags_str,
        'duration_iso': duration_iso,
        'duration_seconds': duration_sec,
        'definition': content.get('definition', ''),
        'caption': content.get('caption', ''),
        'licensed_content': content.get('licensedContent', ''),
        'view_count': safe_int(stats.get('viewCount')),
        'like_count': safe_int(stats.get('likeCount')),
        'dislike_count': safe_int(stats.get('dislikeCount')),
        'comment_count': safe_int(stats.get('commentCount')),
        'favorite_count': safe_int(stats.get('favoriteCount')),
        'classification_label': obj.get('classification_label') or '',
        'prediction': obj.get('prediction') or '',
        'is_ground_truth': is_gt,
        'source_dataset': source_name
    }

FIELDNAMES = [
    'video_id',
    'title',
    'description',
    'channel_id',
    'channel_title',
    'published_at',
    'category_id',
    'tags',
    'duration_iso',
    'duration_seconds',
    'definition',
    'caption',
    'licensed_content',
    'view_count',
    'like_count',
    'dislike_count',
    'comment_count',
    'favorite_count',
    'classification_label',
    'prediction',
    'is_ground_truth',
    'source_dataset'
]

def generate_groundtruth_csv(input_json, output_csv):
    print(f'Generando {output_csv} desde {input_json}...')
    count = 0
    with open(input_json, 'r', encoding='utf-8') as f_in, \
         open(output_csv, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES)
        writer.writeheader()
        for line in f_in:
            if not line.strip():
                continue
            data = json.loads(line)
            row = extract_video_row(data, 'groundtruth')
            writer.writerow(row)
            count += 1
    print(f'Completado: {count} registros en {output_csv}')

def generate_consolidated_csv(file_sources, output_csv):
    print(f'Generando {output_csv} a partir de todos los JSON...')
    seen_ids = set()
    total_written = 0
    total_read = 0

    with open(output_csv, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES)
        writer.writeheader()

        for source_name, file_path in file_sources:
            if not os.path.exists(file_path):
                print(f'Archivo no encontrado: {file_path}')
                continue

            print(f'Procesando {file_path} ({source_name})...')
            file_count = 0
            with open(file_path, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    if not line.strip():
                        continue
                    total_read += 1
                    data = json.loads(line)
                    vid_id = data.get('id')

                    if vid_id and vid_id in seen_ids:
                        continue

                    if vid_id:
                        seen_ids.add(vid_id)

                    row = extract_video_row(data, source_name)
                    writer.writerow(row)
                    total_written += 1
                    file_count += 1

            print(f'Leidos en este archivo: {file_count}')

    print(f'Total lineas procesadas: {total_read}')
    print(f'Total videos unicos exportados: {total_written}')

if __name__ == '__main__':
    base_dir = 'DataSet_Disturbed_YouTube_Kids' if os.path.exists('DataSet_Disturbed_YouTube_Kids') else '.'
    
    gt_json = os.path.join(base_dir, 'groundtruth_videos.json')
    gt_csv = 'dataset_groundtruth.csv'
    generate_groundtruth_csv(gt_json, gt_csv)

    file_sources = [
        ('groundtruth', os.path.join(base_dir, 'groundtruth_videos.json')),
        ('elsagate_related', os.path.join(base_dir, 'elsagate_related_videos.json')),
        ('other_child_related', os.path.join(base_dir, 'other_child_related_videos.json')),
        ('popular', os.path.join(base_dir, 'popular_videos.json')),
        ('random', os.path.join(base_dir, 'random_videos.json'))
    ]
    all_csv = 'dataset_consolidado_completo.csv'
    generate_consolidated_csv(file_sources, all_csv)
