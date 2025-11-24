import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import scipy.optimize
from tabulate import tabulate
import json
import re
from rapidfuzz import process


# load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# use config values
data_path = Path(config['paths']['data'] / config['filenames']['input'])
output_path = Path(config['paths']['output'])

with open('static_data/troops.json', 'r') as t:
    troops = json.load(t)



def get_data() -> pd.DataFrame:
    # Validate config structure11
    required_keys = ['paths', 'filenames', 'column_mapping', 'settings']
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing '{key}' in config.")

    # Build file path
    data_dir = Path(config['paths']['data'])
    filename = config['filenames']['input']
    file_path = data_dir / filename

    # Load data
    if filename.endswith('.xlsx'):
        xls = pd.ExcelFile(file_path)
        sheet_index = config['filenames'].get('data_sheet', 0)
        try:
            sheet_name = xls.sheet_names[sheet_index]
        except IndexError:
            raise ValueError(f"Sheet index {sheet_index} is out of range.")
        df = pd.read_excel(file_path, sheet_name=sheet_name)

    elif filename.endswith('.csv'):
        df = pd.read_csv(file_path)

    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

    # Clean data
    df.fillna(0, inplace=True)
    df.replace('', 0, inplace=True)

    # Rename columns using fuzzy matching
    rename_map = {}
    threshold = config['settings'].get('fuzzy_threshold', 80)
    for original_col in df.columns:
        col_lower = original_col.lower()
        best_score = 0
        best_name = None
        for short_name, keywords in config['column_mapping'].items():
            for keyword in keywords:
                match = process.extractOne(col_lower, [keyword.lower()])
                if match and match[1] > best_score:
                    best_score = match[1]
                    best_name = short_name
        if best_score >= threshold:
            rename_map[original_col] = best_name

    df.rename(columns=rename_map, inplace=True)
    return df

def remove_unused_columns(df: pd.DataFrame, required_columns: list = config['required_columns']) -> pd.DataFrame:
    data = df[required_columns]
    return data

def remove_duplicate_players(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only the row with the most recent completion_time per player
    df["completion_time"] = pd.to_datetime(df["completion_time"], format="%m/%d/%Y %H:%M:%S")
    deduped = df.loc[df.groupby("ID")["completion_time"].idxmax()].reset_index(drop=True)
    return deduped


def fix_times(time: str) -> np.ndarray:
    times = time.replace(' ', '')
    if ',' in times:
        fixed_times = times.split(',')
    elif ';' in times:
        fixed_times = times.split(';')
    else:
        print("Time format invalid. Tried to alter format and failed.")
        fixed_times = []
    return np.array(fixed_times)


def construction_data(data: pd.DataFrame) -> pd.DataFrame:
    data['Points'] = data["RFC"] * 30000 + data['FC'] * 2000 + data["ConstructionSpeed"] * 24 * 60 * 30
    filtered = data[data['Points'] > 0][['Name', 'ID', 'ConstructionTimes', 'Points']]
    return filtered

def research_data(data: pd.DataFrame) -> pd.DataFrame:
    data['Points'] = data['FCS'] * 1000 + data["ResearchSpeed"] * 24 * 60 * 30
    filtered = data[data['Points'] > 0][['Name', 'ID', 'ResearchTimes', 'Points']]
    return filtered

# TODO: find best way to get troop data. potentially enter each type of troop with their current and then desired level


def promotion_errors(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Finds the players who have inncorrectly entered their troop data and information.
    If player has entered the promotion data in the wrong order, their data will not be included in valid players.
    They will not be considered for troop promotion points.
    
    Inputs: data - pd.DataFrame
    
    Returns: (error_players, valid_players) -> data frames
    - error_players: the players who entered data inccorectly or not at all ie. not promoting*
    - valid_players: players with data correctly entered 
    
    
    """
    # TODO: * find a better way to have the error of players who did want to promote only
    
	# points mapping for the troop levels
    data['points_high'] = data["promo_high"].map(troops["points"]).fillna(0)
    data['points_low'] = data["promo_low"].map(troops["points"]).fillna(0)
    
	# point difference calc
    data['point_diff'] = data['points_high'] - data['points_low']
    
   # defining and filtering for the error condition
    error_mask = data['point_diff']<=0
    error_players = data.loc[error_mask, data['Name', 'ID', 'promo_high', 'promo_low', 'troop_bonus', 'promo_no', 'point_diff', 'troop_speed', 'troop_lv']]
    valid_players = data.loc[~error_mask, data['Name', 'ID', 'promo_high', 'promo_low', 'troop_bonus', 'promo_no', 'point_diff', 'troop_speed', 'troop_lv']]
    return error_players, valid_players

# TODO: have an option to list of players that entered the incorrect data


def promos(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_players = promotion_errors(data)[1]

	# Checking for enough speedups
    
    time_map = troops["base_time"] # time map for each troop tier
    
	# point mapping for each troop level
    valid_players['high_time'] = valid_players['promo_high'].map(time_map)
    valid_players['low_time'] = valid_players['promo_low'].map(time_map)
    
	# finding the adjusted time and accounting for troop training bonuses 
    valid_players['adj_time_promo'] = (valid_players['high_time'] - valid_players['low_time'])/(1+(valid_players['troop_bonus']/100))
    
    # time required to promote the listed number of troops
    valid_players['total_time_promo'] = valid_players['adj_time_promo'] * valid_players['promo_no']
    
	# Separates the player who have enough speedups to cover all the promototions with those who do not have enough
    partial_points = valid_players.loc[valid_players['troop_speed'] < valid_players['total_time_promo']]
    full_points = valid_players.loc[valid_players['troop_speed'] >= valid_players['total_time_promo']]
    
	# partial points calculation accounts for only the no. of troop speed availiable
    partial_points['promo_points'] = partial_points['troop_speed']*partial_points['adj_time_promo']*partial_points['point_diff']
    
	# full points 
    full_points['promo_points'] = full_points['point_diff']*full_points['promo_no']
    full_points['troop_speed'] = full_points['troop_speed'] - full_points['total_time_promo']
    
    return partial_points, full_points
    

def training(data: pd.DataFrame) -> pd.DataFrame:
    valid = promos(data)[1]
   




def troop_data(data: pd.DataFrame) -> pd.DataFrame:
    troop_consideration = data[data['troop_speed']>0] # players with more than 0 troop speed if nessecary
    # promotions points
    data['promo_high'] = data["promo_high"].map(troops["points"]).fillna(0)
    data['promo_low'] = data["promo_low"].map(troops["points"]).fillna(0)

    # promotion time w/ no bonus (base time taken to promote 1 troop between the specified levels)
    data['high_time'] = data["promo_high"].map(troops["time"]).fillna(0)
    data['low_time'] = data["promo_low"].map(troops["time"]).fillna(0)
    
	# time for one promotion including bonus
    data['promo_time_w_bonus'] = (data['high_time'] - data['low_time'])/(1+(data['troop_bonus']/100))
    
    # with the remaining speedups this is for the troop training point calculation

    time_promo = data['promo_time_w_speed'] * data['promo_no']
    data['current_speed'] = data['troop_speed'].to_numpy()
    current_speed -= time_promo
    
	leftover_speed = 
        
    


    data['training_points'] = data['troop_lv'].map(troops["points"]).fillna(0)
    data['base_time'] = data['troop_lv'].map(troops["base_time"]).fillna(0)
    data['time_per_troop'] = data['base_time'] / (1 + (data['troop_bonus'] / 100))
    ppm = data['training_points'] / data['time_per_troop']
    data['training_points'] = ppm * current_speed

    return #filtered


def optimize_schedule(data: pd.DataFrame, time_column: str, points_column: str = 'Points', top_n: int = 100) -> pd.DataFrame:
    candidates = data.sort_values(by=points_column, ascending=False).head(top_n).copy()
    candidates.reset_index(drop=True, inplace=True)
    candidates['Rank'] = candidates.index + 1  # Add rank based on points

    time_slots = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
    cost_matrix = np.full((len(time_slots), len(candidates)), 1e6)

    for i, slot in enumerate(time_slots):
        for j, (_, row) in enumerate(candidates.iterrows()):
            available_times = [t.strip() for t in str(row[time_column]).replace(" ", "").replace(";", ",").split(",") if t.strip()]
            if slot in available_times:
                cost_matrix[i, j] = -row[points_column]

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    assignments = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 1e6:
            player = candidates.iloc[c]['Name']
            player_id = candidates.iloc[c]['ID']
            points = candidates.iloc[c][points_column]
            rank = candidates.iloc[c]['Rank']
            assignments.append((rank, time_slots[r], player, player_id, points))

    return pd.DataFrame(assignments, columns=['Rank', 'TimeSlot', 'Player', 'ID', 'Points'])

def export_schedules_to_csv(construction_schedule, research_schedule, troop_schedule, filename):
    output_folder = Path("schedules")
    output_folder.mkdir(parents=True, exist_ok=True)
    construction_schedule.to_csv(output_folder / "monday_construction_schedule.csv", index=False)
    research_schedule.to_csv(output_folder / "tuesday_research_schedule.csv", index=False)
    troop_schedule.to_csv(output_folder / "thursday_troop_schedule.csv", index=False)

def find_unassigned_top_players(full_data: pd.DataFrame, optimized_schedule: pd.DataFrame, time_column: str, top_n: int) -> pd.DataFrame:
    top_players = full_data.sort_values(by='Points', ascending=False).head(top_n).copy()
    top_players.reset_index(drop=True, inplace=True)
    top_players['Rank'] = top_players.index + 1

    assigned_players = set(optimized_schedule['Player'])
    unassigned = top_players[~top_players['Name'].isin(assigned_players)]

    return unassigned[['Rank', 'Name', 'ID', time_column, 'Points']]

def main():
    filename = "mock_schedule_data_extended"  # Replace with actual filename (without .csv)
    data = get_data()
    data = remove_duplicate_players(data)

    # Process each day
    construction_df = construction_data(data.copy())
    research_df = research_data(data.copy())
    troop_df = troop_data(data.copy())

    # Optimize schedules (includes Rank and ID)
    construction_schedule = optimize_schedule(construction_df, time_column='ConstructionTimes', top_n=100)
    research_schedule = optimize_schedule(research_df, time_column='ResearchTimes', top_n=100)
    troop_schedule = optimize_schedule(troop_df, time_column='TroopTimes', top_n=100)

    # Calculate total estimated points
    construction_total = construction_schedule['Points'].sum()
    research_total = research_schedule['Points'].sum()
    troop_total = troop_schedule['Points'].sum()

    # Display optimized schedules with Rank
    print("\n=== MONDAY (Construction Day) ===")
    print(f"Estimated Total Points: {construction_total:,}")
    print(tabulate(construction_schedule.values.tolist(), headers=construction_schedule.columns.tolist(), tablefmt='grid'))

    print("\n=== TUESDAY (Research Day) ===")
    print(f"Estimated Total Points: {research_total:,}")
    print(tabulate(research_schedule.values.tolist(), headers=research_schedule.columns.tolist(), tablefmt='grid'))

    print("\n=== THURSDAY (Troop Day) ===")
    print(f"Estimated Total Points: {troop_total:,}")
    print(tabulate(troop_schedule.values.tolist(), headers=troop_schedule.columns.tolist(), tablefmt='grid'))

    # Export schedules to CSV (includes Rank and ID)
    export_schedules_to_csv(construction_schedule, research_schedule, troop_schedule, filename)

    # Show unassigned top players with Rank
    top_n = 100
    missed_construction = find_unassigned_top_players(construction_df, construction_schedule, 'ConstructionTimes', top_n)
    missed_research = find_unassigned_top_players(research_df, research_schedule, 'ResearchTimes', top_n)
    missed_troop = find_unassigned_top_players(troop_df, troop_schedule, 'TroopTimes', top_n)

    print(f"\n=== UNASSIGNED TOP {top_n} PLAYERS - MONDAY (Construction Day) ===")
    print(tabulate(missed_construction.values.tolist(), headers=missed_construction.columns.tolist(), tablefmt='grid'))

    print(f"\n=== UNASSIGNED TOP {top_n} PLAYERS - TUESDAY (Research Day) ===")
    print(tabulate(missed_research.values.tolist(), headers=missed_research.columns.tolist(), tablefmt='grid'))

    print(f"\n=== UNASSIGNED TOP {top_n} PLAYERS - THURSDAY (Troop Day) ===")
    print(tabulate(missed_troop.values.tolist(), headers=missed_troop.columns.tolist(), tablefmt='grid'))

    # Highlight players who missed multiple days
    names_missed = {
        'Construction': set(missed_construction['Name']),
        'Research': set(missed_research['Name']),
        'Troop': set(missed_troop['Name']),
    }

    repeated_misses = (names_missed['Construction'] & names_missed['Research']) | \
                      (names_missed['Construction'] & names_missed['Troop']) | \
                      (names_missed['Research'] & names_missed['Troop'])

    print("\n=== PLAYERS WHO MISSED MULTIPLE DAYS ===")
    if repeated_misses:
        for name in sorted(repeated_misses):
            missed_days = [day for day, names in names_missed.items() if name in names]
            print(f"- {name} missed: {', '.join(missed_days)}")
    else:
        print("No players missed more than one day.")

if __name__ == "__main__":
    main()
