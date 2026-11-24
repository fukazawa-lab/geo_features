"""Geospatial feature extraction for Google Colab."""

from pathlib import Path
import os, re, zipfile, warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from pyproj import CRS, Transformer
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent
DOWNLOAD = BASE / 'common_data'
EXTRACT = BASE / 'extracted'
CACHE = BASE / 'cache'
for p in (DOWNLOAD, EXTRACT, CACHE): p.mkdir(parents=True, exist_ok=True)
ARCHIVES = ['elevation.zip', 'road.zip', '2024JPN_v25.04.zip']

ELEV_COLS = [
    'mesh_code', 'avg_elevation', 'max_elevation', 'min_elevation',
    'min_elevation_code', 'max_slope_angle', 'max_slope_dir',
    'min_slope_angle', 'min_slope_dir', 'avg_slope_angle'
]
ROAD_CLASSES = ['road25','road19_25','road13_19','road13_unknown',
                'road5_13','road3_5','road_lt3','road_unknown','road_all']
ROAD_COLS = ['mesh_code','mesh_area']
for cls in ROAD_CLASSES:
    ROAD_COLS += [f'{cls}_density_{d}' for d in ['left','top','right','bottom']]
    ROAD_COLS += [f'{cls}_length_total',f'{cls}_length_per_km2']


def mesh_bounds(code):
    code = str(code).strip().removesuffix('.0')
    if len(code) != 8 or not code.isdigit():
        raise ValueError(f'Expected an 8-digit third-level mesh code: {code}')
    lat = int(code[:2]) * (2/3)
    lon = int(code[2:4]) + 100
    lat += int(code[4]) * (2/3/8) + int(code[6]) * (2/3/8/10)
    lon += int(code[5]) * (1/8) + int(code[7]) * (1/8/10)
    return lon, lat, lon + 1/80, lat + 1/120


def find_xml_files(folder, pattern):
    # Expand inner archives in place (original notebooks used per-tile ZIPs).
    for zippath in list(folder.rglob('*.zip')):
        destination = zippath.parent / (zippath.stem + '_unpacked')
        if not destination.exists():
            destination.mkdir(parents=True)
            with zipfile.ZipFile(zippath) as z:
                for item in z.infolist():
                    path = (destination / item.filename).resolve()
                    if not path.is_relative_to(destination.resolve()):
                        raise ValueError('Unsafe nested ZIP path')
                    z.extract(item, destination)
    return sorted(folder.rglob(pattern))


def parse_mesh_xmls(folder, pattern, columns, cache_name):
    cache_file = CACHE / cache_name
    if cache_file.exists():
        return pd.read_pickle(cache_file)
    xml_files = find_xml_files(folder, pattern)
    if not xml_files:
        raise FileNotFoundError(f'No files matching {pattern} in {folder}')
    records = []
    for path in tqdm(xml_files, desc=f'Parsing {pattern}'):
        try:
            # iterparse reduces memory usage for large XML collections.
            for _, elem in ET.iterparse(path, events=('end',)):
                if elem.tag.endswith('}tupleList') or elem.tag == 'tupleList':
                    if elem.text:
                        for t in elem.text.split():
                            v = t.split(',')
                            if len(v) == len(columns):
                                records.append(v)
                    elem.clear()
        except ET.ParseError as e:
            warnings.warn(f'Cannot parse {path}: {e}')
    if not records:
        raise ValueError(f'No matching tuples found for {pattern}')
    df = pd.DataFrame(records, columns=columns)
    df['mesh_code'] = df['mesh_code'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    df = df[df['mesh_code'].str.fullmatch(r'\d{8}')].copy()
    for col in columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.drop_duplicates('mesh_code', keep='first').reset_index(drop=True)
    df.to_pickle(cache_file)
    print(f'{cache_name}: {len(df):,} meshes')
    return df


def load_elevation():
    return parse_mesh_xmls(EXTRACT / 'elevation', 'G04-a-11_*-jgd.xml',
                           ELEV_COLS, 'elevation.pkl')


def load_road():
    return parse_mesh_xmls(EXTRACT / 'road', 'N04-10_*-jgd-g.xml',
                           ROAD_COLS, 'road.pkl')


def mesh_record_features(df_indexed, meshcode, kind):
    """Retrieve original published mesh statistics, without spatial averaging."""
    if kind == 'elevation':
        columns = ['avg_elevation', 'max_elevation', 'min_elevation',
                   'max_slope_angle', 'min_slope_angle', 'avg_slope_angle']
    elif kind == 'road':
        columns = [f'{cls}_length_total' for cls in ROAD_CLASSES]
        columns += [f'{cls}_length_per_km2' for cls in ROAD_CLASSES]
    else:
        raise ValueError(f'Unknown kind: {kind}')

    found = meshcode in df_indexed.index
    result = {'found': found}
    if not found:
        for col in columns:
            renamed = col.replace('_length_total', '_length_total_m').replace(
                '_length_per_km2', '_length_m_per_km2') if kind == 'road' else col
            result[renamed] = np.nan
        return result

    record = df_indexed.loc[meshcode]
    for col in columns:
        renamed = col.replace('_length_total', '_length_total_m').replace(
            '_length_per_km2', '_length_m_per_km2') if kind == 'road' else col
        result[renamed] = record[col]
    return result



import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds, Window
from functools import lru_cache

CATEGORIES = ['Water','Built-up','Paddy','Cropland','Grassland','DBF','DNF',
              'EBF','ENF','Bare','Bamboo','Solar','Wetland','Greenhouse','Rock/Tidal']


@lru_cache(maxsize=1)
def landcover_tifs():
    tif_dir = EXTRACT / '2024JPN_v25.04'
    tifs = tuple(sorted(tif_dir.rglob('LC_N*E*.tif')))
    if not tifs:
        raise FileNotFoundError(f'No GeoTIFF found under {tif_dir}')
    return tifs


@lru_cache(maxsize=1)
def landcover_tile_index():
    """Cache geographic bounds of all land-cover tiles."""
    tiles = []
    for path in landcover_tifs():
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError(f'Raster has no CRS: {path}')
            bounds = transform_bounds(src.crs, 'EPSG:4326', *src.bounds, densify_pts=8)
            tiles.append((path, bounds))
    return tiles


@lru_cache(maxsize=16)
def open_landcover_tif(path):
    return rasterio.open(path)


@lru_cache(maxsize=16)
def get_transformer(crs):
    return Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)


def landcover_proportions(lat, lon, mode='buffer500'):
    """Land-cover fractions within a 500 m circle or the containing 1 km mesh."""
    if mode == 'buffer500':
        lat_delta = 500 / 110_500 + 0.001
        lon_delta = 500 / max(10_000, 111_320*np.cos(np.deg2rad(lat))) + 0.001
        west, south, east, north = lon-lon_delta, lat-lat_delta, lon+lon_delta, lat+lat_delta
    elif mode == 'mesh':
        west, south, east, north = mesh_bounds(latlon_to_meshcode(lat, lon))
    else:
        raise ValueError('mode must be buffer500 or mesh')

    counts = np.zeros(len(CATEGORIES), dtype=np.int64)

    for path, (tw, ts, te, tn) in landcover_tile_index():
        if west >= te or east <= tw or south >= tn or north <= ts:
            continue

        src = open_landcover_tif(path)
        left, bottom, right, top = transform_bounds(
            'EPSG:4326', src.crs, west, south, east, north, densify_pts=8)
        left, right = max(left, src.bounds.left), min(right, src.bounds.right)
        bottom, top = max(bottom, src.bounds.bottom), min(top, src.bounds.top)
        if left >= right or bottom >= top:
            continue

        window = from_bounds(left, bottom, right, top, src.transform)
        window = window.round_offsets().round_lengths()

        full_window = Window(0, 0, src.width, src.height)

        try:
            window = window.intersection(full_window)
        except rasterio.errors.WindowError:
            continue

        if window.width <= 0 or window.height <= 0:
            continue

        arr = src.read(1, window=window, masked=True)

        arr = src.read(1, window=window, masked=True)
        if arr.size == 0:
            continue

        # Vectorized pixel-center coordinates
        t = src.window_transform(window)
        rows, cols = np.indices(arr.shape)
        cols, rows = cols + 0.5, rows + 0.5
        xs = t.a*cols + t.b*rows + t.c
        ys = t.d*cols + t.e*rows + t.f
        if src.crs != rasterio.crs.CRS.from_epsg(4326):
            xs, ys = get_transformer(src.crs.to_string()).transform(xs, ys)

        if mode == 'mesh':
            inside = (xs >= west) & (xs < east) & (ys >= south) & (ys < north)
        else:
            lat1, lat2 = np.deg2rad(lat), np.deg2rad(ys)
            dlat, dlon = lat2-lat1, np.deg2rad(xs-lon)
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            inside = 2*6371000*np.arcsin(np.sqrt(np.clip(a, 0, 1))) <= 500

        valid = inside & ~np.ma.getmaskarray(arr)
        values = np.asarray(arr)[valid]
        values = values[(values >= 1) & (values <= 15)].astype(int)
        counts += np.bincount(values, minlength=16)[1:16]

    total = int(counts.sum())
    return {
        'valid_pixel_count': total,
        **{name: counts[i]/total if total else np.nan
           for i, name in enumerate(CATEGORIES)}
    }


def landcover_in_radius(lat, lon, radius_m=500):
    if radius_m != 500:
        raise ValueError('This notebook uses a fixed 500 m buffer.')
    return landcover_proportions(lat, lon, mode='buffer500')




import io
from functools import lru_cache

CENSUS_ZIP = DOWNLOAD / 'census_2020.zip'


# JIS X 0410 third-level mesh (8 digits).
def latlon_to_meshcode(lat, lon):
    lat1, lon1 = int(lat * 1.5), int(lon - 100)
    lat2, lon2 = int((lat * 60 % 40) / 5), int((lon * 60 % 60) / 7.5)
    lat3, lon3 = int((lat * 60 % 5) * 60 / 30), int((lon * 60 % 7.5) * 60 / 45)
    return f'{lat1:02d}{lon1:02d}{lat2}{lon2}{lat3}{lon3}'


def read_census_tile(code4):
    """Read census TXT files using Shift-JIS and skip the first row."""
    member_name = f'estat_{code4}.zip'
    tables = []
    with zipfile.ZipFile(CENSUS_ZIP) as outer:
        if member_name not in outer.namelist():
            raise FileNotFoundError(f'{member_name} is missing from {CENSUS_ZIP}')
        tile = outer.read(member_name)

    with zipfile.ZipFile(io.BytesIO(tile)) as z:
        for member in z.namelist():
            if member.lower().endswith('.txt'):
                df = pd.read_csv(
                    io.BytesIO(z.read(member)),
                    encoding='shift_jis',
                    skiprows=1,
                    dtype=str,
                    low_memory=False
                )
                df.columns = df.columns.astype(str).str.strip()
                tables.append(df)

    if not tables:
        raise ValueError(f'No census TXT tables in {member_name}')

    return pd.concat(tables, ignore_index=True)


def census_mesh_column(df):
    """Find the 8-digit mesh code column, including unnamed columns."""

    # Check likely identifier columns first.
    candidates = [
        c for c in df.columns
        if 'メッシュ' in c or 'KEY_CODE' in c.upper()
        or 'KEYCODE' in c.upper() or c.startswith('Unnamed:')
    ]

    # Also check remaining columns if necessary.
    candidates += [c for c in df.columns if c not in candidates]

    for col in candidates:
        values = df[col].dropna().astype(str).str.strip()
        values = values.str.replace(r'\.0$', '', regex=True)

        matches = values.str.fullmatch(r'\d{8}')

        if matches.sum() > 0 and matches.mean() > 0.5:
            return col

    raise KeyError(
        'Could not identify the 8-digit mesh code column.\n'
        + df.iloc[:10, :5].to_string(index=False)
    )


@lru_cache(maxsize=32)
def census_tile_index(code4):
    """Build a census table indexed by 8-digit mesh code."""
    df = read_census_tile(code4)
    key = census_mesh_column(df)

    df[key] = (
        df[key].astype(str).str.strip()
        .str.replace(r'\.0$', '', regex=True)
    )

    df = df[df[key].str.fullmatch(r'\d{8}')].copy()
    return df.drop_duplicates(subset=[key]).set_index(key)


def normalize_census_mesh_reference(value):
    """Preserve confidentiality-related mesh references as 8-digit strings."""
    if pd.isna(value):
        return None
    value = str(value).strip()
    value = re.sub(r"\.0$", "", value)
    return value if re.fullmatch(r"\d{8}", value) else None


def census_features(lat, lon):
    """Return census statistics and confidentiality metadata for the containing mesh."""
    mesh = latlon_to_meshcode(lat, lon)
    df = census_tile_index(mesh[:4])
    out = {
        'meshcode': mesh,
        'found': mesh in df.index,
        'confidentiality_code': pd.NA,
        'merged_into_meshcode': None,
        'merged_from_meshcode': None,
    }
    if mesh not in df.index:
        return out

    row = df.loc[mesh]
    # Unnamed: 1: confidentiality class (0: ordinary, 1: aggregation destination,
    # 2: suppressed/aggregated source). Unnamed: 2/3: related mesh identifiers.
    if 'Unnamed: 1' in row.index:
        out['confidentiality_code'] = pd.to_numeric(row['Unnamed: 1'], errors='coerce')
    if 'Unnamed: 2' in row.index:
        out['merged_into_meshcode'] = normalize_census_mesh_reference(row['Unnamed: 2'])
    if 'Unnamed: 3' in row.index:
        out['merged_from_meshcode'] = normalize_census_mesh_reference(row['Unnamed: 3'])

    for col, value in row.items():
        if str(col).startswith('Unnamed:'):
            continue
        value = str(value).strip() if pd.notna(value) else ''
        out[str(col)] = (
            pd.to_numeric(value.replace(',', ''), errors='coerce')
            if value else np.nan
        )
    return out


import re
import unicodedata

def translate_census_column(name):
    s = unicodedata.normalize("NFKC", str(name)).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("〜", "~").replace("～", "~")

    if s in ("meshcode", "found"):
        return s

    # Population by age and sex
    if "人口" in s:
        sex = "_male" if s.endswith("男") else "_female" if s.endswith("女") else "_total"

        if "外国人" in s:
            return "foreign_population" + sex
        if "人口(総数)" in s:
            return "population" + sex

        m = re.search(r"(\d+)~(\d+)歳人口", s)
        if m:
            return f"population_age_{m.group(1)}_{m.group(2)}{sex}"

        m = re.search(r"(\d+)歳以上人口", s)
        if m:
            return f"population_age_{m.group(1)}_plus{sex}"

    # Household statistics
    household_names = {
        "世帯総数": "households_total",
        "一般世帯数": "households_private",
        "親族のみの世帯数 一般世帯数": "households_relatives_only",
        "核家族世帯数 一般世帯数": "households_nuclear_family",
        "核家族以外の世帯数 一般世帯数": "households_non_nuclear_family",
        "6歳未満世帯員のいる世帯数 一般世帯数": "households_with_children_under_6",
        "65歳以上世帯員のいる世帯数 一般世帯数": "households_with_members_65_plus",
        "高齢単身世帯数 一般世帯数": "households_elderly_single",
        "高齢夫婦世帯数 一般世帯数": "households_elderly_couple",
    }

    if s in household_names:
        return household_names[s]

    m = re.fullmatch(r"(\d+)人世帯数 一般世帯数", s)
    if m:
        return f"households_{m.group(1)}_person"

    if s == "7人以上世帯数 一般世帯数":
        return "households_7_plus_persons"

    m = re.fullmatch(
        r"世帯主の年齢が(\d+)~(\d+)歳の1人世帯数 一般世帯数", s
    )
    if m:
        return f"single_person_households_head_age_{m.group(1)}_{m.group(2)}"

    return s

_initialized = False

def initialize():
    global _initialized, elevation_df, road_df, elevation_index, road_index
    if _initialized:
        return
    missing = [x for x in ARCHIVES + [CENSUS_ZIP.name] if not (DOWNLOAD / x).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing ZIP files: {missing}. Download/upload them to {DOWNLOAD}.')

    for filename in ARCHIVES:
        target = EXTRACT / filename.removesuffix('.zip')
        sentinel = target / '.extraction_complete'
        if sentinel.exists():
            print('Already extracted:', filename)
            continue
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(DOWNLOAD / filename) as z:
            for info in tqdm(z.infolist(), desc=f'Extract {filename}'):
                path = (target / info.filename).resolve()
                if not path.is_relative_to(target.resolve()):
                    raise ValueError('Unsafe ZIP path')
                z.extract(info, target)
        sentinel.touch()
        print('Extracted:', filename, '->', target)

    # 初回実行後は解析結果がキャッシュされます。
    elevation_df = load_elevation()
    road_df = load_road()
    elevation_index = elevation_df.set_index('mesh_code', verify_integrity=True)
    road_index = road_df.set_index('mesh_code', verify_integrity=True)


    landcover_tile_index()
    _initialized = True

def extract_features(lat, lon):
    initialize()
    meshcode = latlon_to_meshcode(lat, lon)
    out = {'latitude': lat, 'longitude': lon, 'meshcode': meshcode}
    out.update({'elev_' + k: v for k, v in
                mesh_record_features(elevation_index, meshcode, 'elevation').items()})
    out.update({'road_' + k: v for k, v in
                mesh_record_features(road_index, meshcode, 'road').items()})
    out.update({'lc_buffer500_' + k: v for k, v in
                landcover_proportions(lat, lon, 'buffer500').items()})
    out.update({'lc_mesh_' + k: v for k, v in
                landcover_proportions(lat, lon, 'mesh').items()})
    out.update({'census_' + translate_census_column(k): v
                for k, v in census_features(lat, lon).items()})
    return out



# 一括処理A：メッシュコードのCSV（上の特徴量抽出セルを実行した後に実行）。
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

MESH_INPUT_DIR = EXTRACT / 'meshcode'
MESH_OUTPUT_DIR = BASE / 'output' / 'meshcode'
MESH_INPUT_DIR.mkdir(parents=True, exist_ok=True)
MESH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_third_meshcode(value):
    """Normalize an eight-digit third-level mesh code, including CSV values like 59401148.0."""
    if pd.isna(value):
        return None
    code = str(value).strip()
    if code.endswith('.0'):
        code = code[:-2]
    return code if len(code) == 8 and code.isascii() and code.isdigit() else None


def third_mesh_center(code):
    west, south, east, north = mesh_bounds(code)
    return (south + north) / 2, (west + east) / 2  # latitude, longitude


def enrich_meshcode_files(prefecture='05'):
    """Extract features for unique third-level meshes in a prefecture (01–47)."""
    code = str(prefecture).zfill(2)
    if code not in [f'{i:02d}' for i in range(1, 48)]:
        raise ValueError('Prefecture code must be 01–47')
    archive = DOWNLOAD / 'meshcode.zip'
    with zipfile.ZipFile(archive) as z:
        members = [n for n in z.namelist() if Path(n).name == code + '.csv']
        if len(members) != 1:
            raise ValueError(f'Expected exactly one {code}.csv in {archive}')
        raw = z.read(members[0])
    for encoding in ('utf-8-sig', 'cp932'):
        try:
            source = pd.read_csv(io.BytesIO(raw), encoding=encoding, dtype=str, header=None)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError('Unsupported CSV encoding')
    candidates = [c for c in source.columns if source[c].str.strip().str.fullmatch(r'\d{4}[0-7]{2}\d{2}[*＊]?').fillna(False).sum() > 0]
    if len(candidates) != 1:
        raise ValueError('Cannot identify unique meshcode column')
    values = source[candidates[0]].str.strip().str.rstrip('*＊')
    values = values[values.str.fullmatch(r'\d{4}[0-7]{2}\d{2}').fillna(False)].drop_duplicates()
    if values.empty:
        raise ValueError('No valid mesh codes')
    file = MESH_INPUT_DIR / f'{code}.csv'
    pd.DataFrame({'meshcode': values}).to_csv(file, index=False, encoding='utf-8-sig')
    files = [file]
    if not files:
        raise FileNotFoundError(f'No CSV files in {MESH_INPUT_DIR}')
    # Each mesh has one center and one fixed feature vector, even across input files.
    feature_cache = {}
    for file in files:
        df = pd.read_csv(file, dtype={'meshcode': 'string'}, encoding='utf-8-sig')
        if 'meshcode' not in df.columns:
            print(f'SKIP {file.name}: missing meshcode column')
            continue
        enriched = []
        for raw_code in tqdm(df['meshcode'], desc=f'Mesh: {file.stem}'):
            code = normalize_third_meshcode(raw_code)
            if code is None:
                enriched.append({'feature_error': 'Invalid or missing third-level meshcode'})
                continue
            if code not in feature_cache:
                lat, lon = None, None
                try:
                    lat, lon = third_mesh_center(code)
                    feat = extract_features(lat, lon)
                    if feat['meshcode'] != code:
                        raise ValueError(f'Mesh mismatch: expected {code}, got {feat["meshcode"]}')
                    feature_cache[code] = {**feat, 'feature_error': ''}
                except Exception as exc:
                    feature_cache[code] = {
                        'latitude': lat,
                        'longitude': lon,
                        'feature_error': str(exc)
                    }
            enriched.append(feature_cache[code])
        # Avoid duplicate column names if a source CSV already has feature columns.
        extra = pd.DataFrame(enriched, index=df.index).drop(columns=df.columns, errors='ignore')
        result = pd.concat([df, extra], axis=1)
        destination = MESH_OUTPUT_DIR / f'{file.stem}_features.csv'
        result.to_csv(destination, index=False, encoding='utf-8-sig')
        print(f'Saved {destination} ({len(result):,} rows, {len(result.columns)} columns)')




# 一括処理B：任意の観測地点のCSV（一括処理Aとは独立して実行可能）。
POINT_INPUT_DIR = BASE / 'research_data'
POINT_OUTPUT_DIR = BASE / 'output' / 'points'
POINT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
POINT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def enrich_point_files():
    files = sorted(POINT_INPUT_DIR.glob('*.csv'))
    if not files:
        raise FileNotFoundError(f'No CSV files in {POINT_INPUT_DIR}')
    for file in files:
        df = pd.read_csv(file, encoding='utf-8-sig', low_memory=False)
        if not {'lat', 'lon'}.issubset(df.columns):
            print(f'SKIP {file.name}: lat/lon columns required')
            continue
        enriched = []
        for p in tqdm(df.itertuples(index=False), total=len(df), desc=f'Points: {file.stem}'):
            lat, lon = pd.to_numeric(p.lat, errors='coerce'), pd.to_numeric(p.lon, errors='coerce')
            if pd.isna(lat) or pd.isna(lon) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
                enriched.append({'feature_error': 'Invalid or missing latitude/longitude'})
                continue
            try:
                enriched.append({**extract_features(float(lat), float(lon)), 'feature_error': ''})
            except Exception as exc:
                enriched.append({'feature_error': str(exc)})
        extra = pd.DataFrame(enriched, index=df.index).drop(columns=df.columns, errors='ignore')
        result = pd.concat([df, extra], axis=1)
        destination = POINT_OUTPUT_DIR / f'{file.stem}_features.csv'
        result.to_csv(destination, index=False, encoding='utf-8-sig')
        print(f'Saved {destination} ({len(result):,} rows, {len(result.columns)} columns)')




import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from branca.colormap import LinearColormap

def plot_mesh_feature(area_name, feature_column='elev_avg_elevation',
                      cmap_name='terrain', save_html=True):

    # Load mesh-level features
    path = BASE / 'output' / 'meshcode' / f'{area_name}_features.csv'
    df = pd.read_csv(path, dtype={'meshcode': 'string'})

    if feature_column not in df.columns:
        raise ValueError(
            f'Column not found: {feature_column}\n'
            f'Available columns: {df.columns.tolist()}'
        )

    df['value'] = pd.to_numeric(df[feature_column], errors='coerce')
    df = df.dropna(subset=['meshcode']).drop_duplicates('meshcode')

    valid = df['value'].dropna()
    if valid.empty:
        raise ValueError(f'No valid values for {feature_column}')

    # Robust color scale: 2nd–98th percentile
    vmin, vmax = valid.quantile([0.02, 0.98])
    if vmin == vmax:
        vmax = vmin + 1

    cmap = plt.get_cmap(cmap_name)

    def get_color(value):
        if pd.isna(value):
            return '#aaaaaa'
        ratio = np.clip((value - vmin) / (vmax - vmin), 0, 1)
        return mcolors.to_hex(cmap(0.15 + 0.8 * ratio))

    # Calculate map center from all mesh bounds
    bounds_list = []
    for code in df['meshcode']:
        try:
            bounds_list.append(mesh_bounds(str(code)))
        except ValueError:
            bounds_list.append(None)

    df['bounds'] = bounds_list
    df = df[df['bounds'].notna()].copy()

    west = min(b[0] for b in df['bounds'])
    south = min(b[1] for b in df['bounds'])
    east = max(b[2] for b in df['bounds'])
    north = max(b[3] for b in df['bounds'])

    m = folium.Map(
        location=[(south + north) / 2, (west + east) / 2],
        zoom_start=9,
        tiles=None,
        prefer_canvas=True
    )

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Tiles © Esri',
        name='Esri World Topographic Map',
        overlay=False,
        control=True
    ).add_to(m)

    layer = folium.FeatureGroup(name=feature_column)

    for _, row in df.iterrows():
        w, s, e, n = row['bounds']
        value = row['value']

        folium.Rectangle(
            bounds=[[s, w], [n, e]],
            color='#333333',
            weight=0.2,
            fill=True,
            fill_color=get_color(value),
            fill_opacity=0.75,
            popup=(
                f"meshcode: {row['meshcode']}<br>"
                f"{feature_column}: "
                f"{value:.3f}" if pd.notna(value) else
                f"meshcode: {row['meshcode']}<br>{feature_column}: No data"
            )
        ).add_to(layer)

    layer.add_to(m)

    colorbar = LinearColormap(
        colors=[mcolors.to_hex(cmap(x)) for x in np.linspace(0.15, 0.95, 7)],
        vmin=float(vmin),
        vmax=float(vmax),
        caption=feature_column
    )
    colorbar.add_to(m)

    m.fit_bounds([[south, west], [north, east]])

    if save_html:
        output = BASE / 'output' / f'{area_name}_{feature_column}_map.html'
        output.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output))
        print(f'Saved: {output}')

    return m

def extract_points(points):
    rows = []
    for p in tqdm(points.itertuples(index=False), total=len(points)):
        rows.append({**p._asdict(), **extract_features(p.latitude, p.longitude)})
    features = pd.DataFrame(rows)
    output_path = BASE / 'output' / 'point_mesh_features.csv'
    features.to_csv(output_path, index=False, encoding='utf-8-sig')
    return features
