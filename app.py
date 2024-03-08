import tkinter as tk
from tkinter import ttk
import pandas as pd

def calculate_strength_for_addmix(df,grade,add_mix_name, base_strength):
    """
    Calculate the increased strength for all curing days for a given add_mix and base_strength.
    
    Parameters:
    - df: DataFrame containing the add_mix data.
    - add_mix_name: Name of the add_mix to filter on.
    - base_strength: Base strength of the material.
    
    Returns:
    - DataFrame with curing days and the calculated increased strengths for the specified add_mix.
    """
    # Filter the DataFrame for the specified add_mix
    filtered_df = df[df['add_mix'] == add_mix_name]
    
    # Calculate the increased strength for each row
    increased_strengths = []
    for _, row in filtered_df.iterrows():
        increase = base_strength * (1 + row['strength_increase_prec'] / 100)
        increased_strengths.append((row['curing_days'], increase))
    
    # Convert the list of tuples to a DataFrame
    result_df = pd.DataFrame(increased_strengths, columns=['Curing Days', 'Increased Strength'])
    
    print(f"Calculated increased strengths for addition of {add_mix_name} at optimum percentage by weight of {filtered_df['add_mix_percent'][0]}% in Grade {grade}:")
    for _, row in result_df.iterrows():
        print(f"Curing days: {row['Curing Days']}, Increased strength: {row['Increased Strength']}%")    
        

df = pd.read_csv('data_all.csv') # Load the data from the CSV file

def calculate_strength_for_addmix_gui(grade, add_mix_name, base_strength):
    """
    Calculate the increased strength for a given additive mixture (add_mix) and base strength.

    Parameters:
    - grade (str): The grade of the material.
    - add_mix_name (str): The name of the additive mixture.
    - base_strength (float): The base strength of the material.

    Returns:
    - result_df (pandas.DataFrame): A DataFrame containing the increased strengths for different curing days.
    - optimum_percent (float): The optimum percentage value for the additive mixture.

    """
    filtered_df = df[df['add_mix'] == add_mix_name]
    increased_strengths = []
    for _, row in filtered_df.iterrows():
        increase = base_strength * (1 + row['strength_increase_prec'] / 100)
        increased_strengths.append((row['curing_days'], increase))
    result_df = pd.DataFrame(increased_strengths, columns=['Curing Days', 'Increased Strength'])
    # Get the first (or any) optimum percentage value for the add_mix
    optimum_percent = filtered_df['add_mix_percent'].iloc[0]
    return result_df, optimum_percent

def on_calculate():
    """
    Perform calculations and display results based on user inputs.

    This function retrieves the values entered by the user for the mix name, base strength, and grade.
    It then calls the calculate_strength_for_addmix_gui function to calculate the strength results.
    The results are displayed in the result_text widget.

    Parameters:
    None

    Returns:
    None
    """
    add_mix_name = add_mix_var.get()
    base_strength = float(base_strength_entry.get())
    grade = grade_entry.get()  # Get grade from entry widget
    result_df,o_p= calculate_strength_for_addmix_gui(grade, add_mix_name, base_strength)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Results for adding {add_mix_name} to Cement of grade {grade} with Target strength {base_strength} at at optimum percentage by weight of {o_p}:\n")
    for _, row in result_df.iterrows():
        result_text.insert(tk.END, f"Curing days: {row['Curing Days']}, Increased strength: {row['Increased Strength']}\n")

# Setup the main window
root = tk.Tk()
root.title("Strength Increase Calculator")

# Dropdown for add_mix
add_mix_var = tk.StringVar()
add_mix_label = ttk.Label(root, text="Select Add Mix:")
add_mix_label.pack()
add_mix_dropdown = ttk.Combobox(root, textvariable=add_mix_var)
add_mix_dropdown['values'] = [
    'Crushed sea shell (FA)',
    'Metakaolin',
    'Quartz powder (FA)',
    'Steel slag',
    'Alccofine',
    'Polypropylene fibres',
    'Silica fume',
    'Marble powder (FA)',
    'Recron 3s fibres',
    'Titanium dioxide',
    'GGBS',
    'Graphene oxide',
    'Zeolite powder (FA)',
    'Bamboo fibres',
    'Hyposludge',
    'Zeolite powder',
    'Hooked steel fibres',
    'Dunite powder',
    'AR Glass fibres',
    'Chebula powder',
    'Crumb rubber (CA)',
    'HDPE',
    'Robo sand (FA)',
    'Copper slag (FA)',
    'Waste foundry sand (FA)',
    'Microsilica',
    'POFA',
    'Bamboo chips (CA)',
    'Bacteria subtilis',
    'M-sand (FA)',
    'Gold mine tailings (FA)',
    'Sulphur powder',
    'Baggase Ash (FA)',
    'Abca fibres ',
    'Kenaf fibres',
    'Crimped steel fibres',
    'Dolomite',
    'Groundnut shell ash (FA)',
    'Hemp fibres',
    'Glass powder (FA)',
    'Biocement',
    'Pine Apple leaf fibres',
    'Jute fibres',
    'Calcium chloride',
    'Coir fibres'
    ] 
add_mix_dropdown.pack()

# Entry for base strength
base_strength_label = ttk.Label(root, text="Enter Target Strength:")
base_strength_label.pack()
base_strength_entry = ttk.Entry(root)
base_strength_entry.pack()

# Entry for grade
grade_label = ttk.Label(root, text="Enter Grade:")
grade_label.pack()
grade_entry = ttk.Entry(root)  # Entry widget for grade
grade_entry.pack()

# Button to calculate
calculate_button = ttk.Button(root, text="Calculate", command=on_calculate)
calculate_button.pack()

# Text area with scrollbar for results
result_frame = tk.Frame(root)
scrollbar = tk.Scrollbar(result_frame)
result_text = tk.Text(result_frame, height=10, width=50, yscrollcommand=scrollbar.set)
scrollbar.config(command=result_text.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
result_frame.pack(fill=tk.BOTH, expand=True)

root.mainloop()