import json
import os
import warnings

from tqdm import tqdm

import mlflow
from datasets import load_dataset

warnings.filterwarnings("ignore")

# Configuration
MLFLOW_URI = "http://127.0.0.1:5000"
HUGGINGFACE_DATASET = "lmarena-ai/arena-hard-auto"
# MLFLOW_DATASET_NAME = "arena_hard_auto"
# EXPERIMENT_NAME = "DATASET_Arena_Hard"
MLFLOW_DATASET_NAME = "arena_hard_v2_0"
EXPERIMENT_NAME = "DATASET_Arena_Hard_V2"

# Setup MLflow
mlflow.set_tracking_uri(MLFLOW_URI)


def create_or_get_experiment(experiment_name):
    """Create experiment if it doesn't exist, otherwise return existing."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✅ Created new experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"📂 Using existing experiment: {experiment_name} (ID: {experiment_id})")
    return experiment_id


def load_arena_hard_dataset(max_samples=None):
    """Load Arena Hard dataset using specific data file."""
    print(f"\n🔄 Loading Arena Hard dataset from HuggingFace")

    try:
        # Load only the question file which has the prompts/questions
        dataset = load_dataset(
            HUGGINGFACE_DATASET,
            # data_files="data/arena-hard-v0.1/question.jsonl",
            data_files="data/arena-hard-v2.0/question.jsonl",
            split="train"
        )

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        print(f"✅ Loaded {len(dataset)} questions from Arena Hard")
        return dataset

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\n💡 Trying alternative loading method...")

        try:
            # Alternative: Load all data and filter
            dataset = load_dataset(
                HUGGINGFACE_DATASET,
                split="train"
            )

            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))

            print(f"✅ Loaded {len(dataset)} records from Arena Hard")
            return dataset

        except Exception as e2:
            print(f"❌ Alternative method also failed: {e2}")
            raise


def convert_to_mlflow_records(hf_dataset):
    """
    Convert Arena Hard dataset to MLflow record format.

    Arena Hard structure:
    - question_id: unique identifier
    - turns: list of conversation turns
    - category: question category
    - reference: optional reference answer
    """
    print("\n🔄 Converting to MLflow record format...")

    mlflow_records = []

    for idx, item in enumerate(tqdm(hf_dataset, desc="Converting records")):
        try:
            # Extract question - handle both single turn and multi-turn
            if "turns" in item and item["turns"]:
                if isinstance(item["turns"], list):
                    question = item["turns"][0] if len(item["turns"]) > 0 else ""
                else:
                    question = str(item["turns"])
            else:
                # Fallback: check for other question fields
                question = (
                        item.get("question", "") or
                        item.get("prompt", "") or
                        item.get("input", "")
                )

            if not question:
                print(f"⚠️ Warning: No question found in record {idx}, skipping")
                continue

            # Build the record
            record = {
                "inputs": {
                    "question": question,
                    "question_id": str(item.get("question_id", f"arena_hard_{idx}")),
                },
                "tags": {
                    "question_id": str(item.get("question_id", f"arena_hard_{idx}")),
                    "source": "arena-hard-auto",
                    "dataset_version": "v0.1"
                }
            }

            # Add category/cluster if available
            category = item.get("category") or item.get("cluster")
            if category:
                record["inputs"]["category"] = str(category)
                record["tags"]["category"] = str(category)

            # Add reference answer as expectation if available
            reference = item.get("reference")
            if reference:
                record["expectations"] = {
                    "reference_answer": str(reference)
                }

            # Add multi-turn conversation if available
            if "turns" in item and isinstance(item["turns"], list) and len(item["turns"]) > 1:
                record["inputs"]["turns"] = item["turns"]
                record["tags"]["multi_turn"] = "true"

            mlflow_records.append(record)

        except Exception as e:
            print(f"⚠️ Warning: Error processing record {idx}: {e}")
            continue

    print(f"✅ Converted {len(mlflow_records)} records")
    return mlflow_records


def create_mlflow_dataset(dataset_name, experiment_id, records, tags=None):
    """Create MLflow dataset and add records."""
    from mlflow.genai.datasets import create_dataset, search_datasets

    # Check if dataset already exists
    print(f"\n🔍 Checking if dataset '{dataset_name}' already exists...")
    existing_datasets = search_datasets(
        experiment_ids=[experiment_id],
        filter_string=f"name = '{dataset_name}'"
    )

    if existing_datasets:
        print(f"📂 Found existing dataset: {dataset_name}")
        dataset = existing_datasets[0]
        #print(f"   Current records: {len(dataset.records)}")

        # Ask user if they want to add to existing or create new
        response = input("\n⚠️  Dataset already exists. Options:\n"
                         "   [1] Add new records to existing dataset\n"
                         "   [2] Delete and recreate dataset\n"
                         "   [3] Cancel\n"
                         "Choose (1/2/3): ").strip()

        if response == "2":
            from mlflow.genai.datasets import delete_dataset
            delete_dataset(dataset_id=dataset.dataset_id)
            print(f" Deleted existing dataset")
            dataset = None
        elif response == "3":
            print("❌ Operation cancelled")
            return None
        # If response == "1", continue with existing dataset
    else:
        dataset = None

    # Create new dataset if needed
    if dataset is None:
        print(f"\n🔨 Creating new MLflow dataset: {dataset_name}")

        default_tags = {
            "source": "huggingface",
            "hf_dataset": HUGGINGFACE_DATASET,
            "version": "1.0",
            "status": "active"
        }

        if tags:
            default_tags.update(tags)

        dataset = create_dataset(
            name=dataset_name,
            experiment_id=[experiment_id],
            tags=default_tags
        )
        print(f"✅ Created dataset: {dataset.dataset_id}")

    # Add records in batches
    print(f"\n📥 Adding {len(records)} records to dataset...")

    BATCH_SIZE = 50  # Reduced batch size for stability
    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Adding batches"):
        batch = records[i:i + BATCH_SIZE]
        try:
            dataset.merge_records(batch)
        except Exception as e:
            print(f"\n⚠️ Warning: Error adding batch {i // BATCH_SIZE + 1}: {e}")
            # Try adding records one by one in this batch
            for record in batch:
                try:
                    dataset.merge_records([record])
                except Exception as e2:
                    print(f"⚠️ Failed to add record: {e2}")

    print(f"✅ Successfully processed all records")
    #print(f"   Total records in dataset: {len(dataset.records)}")

    return dataset


def main():
    """Main execution function."""
    print("=" * 80)
    print("🚀 Arena Hard Dataset - HuggingFace to MLflow Ingestion")
    print("=" * 80)

    # Step 1: Create or get experiment
    experiment_id = create_or_get_experiment(EXPERIMENT_NAME)

    # Step 2: Load Arena Hard dataset
    print("\n💡 Note: Loading Arena Hard question set...")
    print("   This dataset contains challenging prompts for LLM evaluation")

    hf_dataset = load_arena_hard_dataset(
        # max_samples=10  # Uncomment to test with small sample
    )

    print(f"\n📊 Dataset Info:")
    print(f"   Name: {HUGGINGFACE_DATASET}")
    print(f"   Total samples: {len(hf_dataset)}")
    print(f"   Features: {list(hf_dataset.features.keys())}")

    # Show sample record
    if len(hf_dataset) > 0:
        print(f"\n📝 Sample record (first question):")
        sample = hf_dataset[0]
        for key, value in sample.items():
            # Truncate long values for display
            display_value = str(value)[:200] + "..." if len(str(value)) > 200 else value
            print(f"   {key}: {display_value}")

    # Step 3: Convert to MLflow format
    mlflow_records = convert_to_mlflow_records(hf_dataset)

    if not mlflow_records:
        print("❌ No records were converted. Please check the dataset structure.")
        return

    # Step 4: Create MLflow dataset
    dataset = create_mlflow_dataset(
        dataset_name=MLFLOW_DATASET_NAME,
        experiment_id=experiment_id,
        records=mlflow_records,
        tags={
            "domain": "general-qa",
            "difficulty": "hard",
            "source_platform": "lmarena",
            "benchmark": "arena-hard-v0.1"
        }
    )

    if dataset:
        print("\n" + "=" * 80)
        print("✅ SUCCESS - Dataset Ingestion Complete!")
        print("=" * 80)
        print(f"📊 Dataset Summary:")
        print(f"   Name: {dataset.name}")
        print(f"   ID: {dataset.dataset_id}")
        #print(f"   Records: {len(dataset.records)}")
        print(f"   Tags: {dataset.tags}")
        print(f"\n🔗 View in MLflow UI:")
        print(f"   {MLFLOW_URI}")

        # Convert to DataFrame and show sample
        print(f"\n📋 Sample Records:")
        df = dataset.to_df()
        print(df.head(3))

        print(f"\n💡 Next Steps:")
        print(f"   1. Review the dataset in MLflow UI")
        print(f"   2. Use this dataset for evaluation with mlflow.genai.evaluate()")
        print(f"   3. Add your model predictions and compare with reference answers")
    else:
        print("\n❌ Dataset ingestion was cancelled or failed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback

        traceback.print_exc()