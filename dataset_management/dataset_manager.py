import os
import json

from dataset_management.export.csv_exporter import CSVExporter
from dataset_management.repair.file_repairer import FileRepairer

class DatasetManager:
    def __init__(self, data_directory, save_path, dataset, appliance_name, debug=False, max_num_houses = None, max_num_rows = 1 * (10**6)):
        self.debug = debug
        self.data_directory = data_directory
        self.save_path = save_path
        self.dataset = dataset.lower()
        self.appliance_name = appliance_name.lower()
        self.appliance_name_formatted = self.appliance_name.replace(" ", "_") 
        self.max_num_houses = max_num_houses

        self.houses = self.getHouses()
        if not self.houses:
            raise ValueError(f"{self.appliance_name} not found in dataset {self.dataset}")

        self.max_num_rows = max_num_rows

        self.house_data_map = self.loadData()
        self.repairer = FileRepairer()
        self.exporter = CSVExporter()
    
    def getHouses(self):
        """
        Get houses for a given dataset and appliance.
        """
        appliance_mappings_dir = os.path.join("dataset_management","data_separation", f"{self.dataset}_appliance_mappings.json")
        houses = []
        if not os.path.exists(appliance_mappings_dir):
            for name in sorted(os.listdir(self.data_directory)):
                if not name.startswith("House_"):
                    continue
                house_number = int(name.split("_", 1)[1])
                appliance_file = os.path.join(
                    self.data_directory,
                    name,
                    f"{self.appliance_name_formatted}_H{house_number}.h5",
                )
                if os.path.exists(appliance_file):
                    houses.append(house_number)
            return houses
        with open(appliance_mappings_dir, 'r') as f:
            appliance_mappings = json.load(f)
        for house in appliance_mappings:
            appliance_list = appliance_mappings[house].values()
            # Check if appliance is present in the house
            if self.appliance_name_formatted in appliance_list:
                house_number = int(house.split(" ")[1])
                houses.append(house_number)
            # Stop if we have reached the maximum number of houses
            if self.max_num_houses:
                if len(houses) == self.max_num_houses:
                    break
        if self.debug:
            print(f"Using houses: {houses} for appliance {self.appliance_name}")
        return houses

    def loadData(self):
        """
        Load data file paths for all houses.
        """
        house_data_map = {}

        for house in self.houses:
            house_dir = f"House_{house}"
            house_path = os.path.join(self.data_directory, house_dir)
            aggregate_file = os.path.join(house_path, f'aggregate_H{house}.h5')
            appliance_file = os.path.join(house_path, f'{self.appliance_name_formatted}_H{house}.h5')

            if not os.path.exists(aggregate_file) or not os.path.exists(appliance_file):
                continue
            house_data_map[house] = {
                "aggregate_file": aggregate_file,
                "appliance_file": appliance_file,
            }
        return house_data_map

    def createData(self):
        repaired_root = os.path.join(self.save_path, "_repaired_cache")
        for house in self.houses:
            if house not in self.house_data_map:
                continue
            os.makedirs(self.save_path, exist_ok=True)
            house_dir = os.path.join(repaired_root, f"House_{house}")
            os.makedirs(house_dir, exist_ok=True)

            aggregate_input = self.house_data_map[house]["aggregate_file"]
            appliance_input = self.house_data_map[house]["appliance_file"]
            aggregate_repaired = os.path.join(house_dir, f"aggregate_H{house}.h5")
            appliance_repaired = os.path.join(house_dir, f"{self.appliance_name_formatted}_H{house}.h5")

            self.repairer.repair_file(aggregate_input, self.dataset, aggregate_repaired)
            self.repairer.repair_file(appliance_input, self.dataset, appliance_repaired)

            output_file = os.path.join(self.save_path, f'{self.appliance_name_formatted}_H{house}.csv')
            self.exporter.export_house_appliance(
                aggregate_repaired,
                appliance_repaired,
                self.dataset,
                output_file,
            )

        


