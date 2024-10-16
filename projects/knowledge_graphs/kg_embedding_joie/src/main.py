from Trainer import Trainer
import argparse


def main():
    parser = argparse.ArgumentParser(description="Training FUSE for Knowledge Graph Embedding")
    # fuzzy set modeling related
    parser.add_argument("--entity_embed_dim", type=int, default=512)
    parser.add_argument("--n_partitions", type=int, default=400)
    parser.add_argument("--regularizer", type=str, default="sigmoid")
    parser.add_argument("--n_rel_basis", type=int, default=50)
    parser.add_argument("--partition_reg_type", type=str, default="sigmoid")
    parser.add_argument("--modulelist", action="store_true")
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=50)
    parser.add_argument("--use_volume_weights", action="store_true")
    parser.add_argument("--regularize_volume", action="store_true")
    parser.add_argument("--score_type", type=str, default="cosine")
    parser.add_argument("--dropout", type=float, default=0.3)

    
    # resource related
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--cpu_num", type=int, default=10)
    
    # paths
    parser.add_argument("--from_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/")
    parser.add_argument("--kg_file_path", type=str, default="data/dbpedia/db_kg_mini.txt")
    parser.add_argument("--kg_ontology_path", type=str, default="data/dbpedia/db_InsType_mini.txt")
    parser.add_argument("--kg_train_file_path", type=str, default="data/dbpedia/db_kg_train.txt")
    parser.add_argument("--kg_train_ontology_path", type=str, default="data/dbpedia/db_InsType_train.txt")
    parser.add_argument("--kg_testfile_path", type=str, default="data/dbpedia/db_kg_test.txt")
    parser.add_argument("--kg_test_ontology_path", type=str, default="data/dbpedia/db_InsType_test.txt")    
    
    
    # training configurations    
    parser.add_argument("--kg_batchsize", type=int, default=2056)
    parser.add_argument("--test_batch_size", default=4, type=int)
    parser.add_argument("--n_negative_samples", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--every_n_steps_to_log", type=int, default=5)
    parser.add_argument("--every_n_steps_to_save", type=int, default=10000)
    parser.add_argument("--every_n_steps_scheduler_update", type=int, default=5000)
    parser.add_argument("--valid_steps", default=10, type=int) # debug with smog test
    parser.add_argument("--warm_up_steps", default=None, type=int)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin_p", type=float, default=0)
    parser.add_argument("--margin_n", type=float, default=0)
    parser.add_argument("--uni_weight", action="store_true")
    parser.add_argument("--scheduler", type=str, default="multistep")

    # loss weights
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--scale", type=int, default=20)


    args = parser.parse_args()
    trainer = Trainer(args)
    print(f"configuration: {args}")

    args = parser.parse_args()
    trainer = Trainer(args)
    print("start training...")
    trainer.train()
    print("finished training: evaluate...")
    trainer.evaluate()
    print("DONE")

if __name__ == "__main__":
    main()