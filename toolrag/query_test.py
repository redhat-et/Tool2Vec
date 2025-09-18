from toolrefiner.query_tool_selector import load_model_and_data, select_tools_for_query

if __name__ == "__main__":
    # Example arguments (update these paths as needed)
    model_name = "microsoft/deberta-v3-base"
    checkpoint_path = "toolrefiner/checkpoints/model_epoch_1.pt"
    tool_embeddings_path = "output/test_embeddings_embedded.pkl"
    tool_names_path = "test.json"
    tool_embedding_dim = 768
    query = "I am currently tracking a package with the ID CA107308006SI. Can you provide me with the latest information and localization details of the package?"

    print(f"Loading model and data...")
    model, tokenizer, tool_embeddings, tool_names = load_model_and_data(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        tool_embeddings_path=tool_embeddings_path,
        tool_names_path=tool_names_path,
        tool_embedding_dim=tool_embedding_dim
    )

    print(f"Selecting tools for query: {query}")
    selected_tools = select_tools_for_query(
        query=query,
        model=model,
        tokenizer=tokenizer,
        tool_embeddings=tool_embeddings,
        tool_names=tool_names,
        top_k=5,
        threshold=0.5,
        save_results=False
    )

    print("\nSelected tools:")
    for tool in selected_tools:
        print(f"  {tool['rank']}. {tool['tool_name']} (score: {tool['score']:.4f})")
