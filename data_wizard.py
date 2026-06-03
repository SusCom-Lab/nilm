from dataset_management.data_separation.data_separator import DataSeparator
from dataset_management.pipeline.pipeline_runner import PipelineRunner
import sys


DATASET_TYPES = ("ukdale", "redd", "refit", "eco", "standard_h5")


def runDataSeparator():
    print("Data Separator")

    file_path = input("Enter the file path to the raw data: ")

    save_path = input("Enter the save path of the separated data: ")

    num_houses = input("Enter the maximum number of houses to process (press enter to process all): ")
    try:
        num_houses = int(num_houses)
    except ValueError:
        num_houses = None
    
    dataset_type = input("Enter the dataset type: ")
    if dataset_type.lower() not in DATASET_TYPES:
        print("Invalid dataset type. Please try again.")
        return
    
    appliance = input("Enter the appliance to filter for (press enter to process all): ")
    if not appliance.strip():
        appliance = None

    data_separator = DataSeparator(file_path=file_path, save_path=save_path, dataset_type=dataset_type, appliance_name=appliance, num_houses=num_houses)
    data_separator.process_data()


def runFullDataPipeline():
    print("Full Data Pipeline")

    input_path = input("Enter the input path (raw dataset path or STANDARD_H5 file path): ")
    workspace_dir = input("Enter the workspace/output directory: ")
    dataset_type = input("Enter the dataset type: ")

    if dataset_type.lower() not in DATASET_TYPES:
        print("Invalid dataset type. Please try again.")
        return

    runner = PipelineRunner()
    summary = runner.run(dataset_type=dataset_type, input_path=input_path, workspace_dir=workspace_dir)

    print("Pipeline completed.")
    print(f"Separated data: {summary['separated_dir']}")
    print(f"Repaired data: {summary['repaired_dir']}")
    print(f"Exported CSVs: {summary['exported_dir']}")
    print(f"Raw validation report: {summary['validation_raw_report']}")
    print(f"Repaired validation report: {summary['validation_repaired_report']}")
    print(f"Exported file count: {summary['export_count']}")


def main():
    while True:
        print("Select an option:")
        print("1. Run Data Separator")
        print("2. Run Full Data Pipeline")
        print("3. Exit")
        
        choice = input("Enter your choice (1/2/3): ")
        
        if choice == '1':
            runDataSeparator()
        elif choice == '2':
            runFullDataPipeline()
        elif choice == '3':
            print("Exiting...")
            sys.exit()
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
