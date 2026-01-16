"""
Create a clinician-friendly decision tree for MIMIC-IV ARDS prediction.
The tree is trained on patX pattern features selected by SHAP importance
from a LightGBM model (max_depth=4 for the tree).
"""
import os
import sys
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Add eval_scripts to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "eval_scripts"))
from core import PatternExtractor
from models import LightGBMWrapper


def load_mimic_data():
    """Load MIMIC data directly and binarize ARDS."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    df = pd.read_csv(
        os.path.join(base_dir, "processed_datasets", "mimic", "mimic_processed.csv")
    )
    y = (df["ARDS_FLAG"] > 0).astype(int)
    feature_cols = [
        col for col in df.columns
        if col not in ["subject_id", "anchor_age", "ARDS_FLAG"]
    ]
    series_names = sorted(
        list(set(col.split("_hour_")[0] for col in feature_cols if "_hour_" in col))
    )
    X_list = [
        df[
            sorted(
                [c for c in feature_cols if c.startswith(f"{s}_hour_")],
                key=lambda x: int(x.split("_hour_")[1]),
            )
        ]
        for s in series_names
    ]
    return X_list, y, series_names


def load_patterns_from_json(json_path):
    """Load patterns from fold_1 of the JSON file."""
    with open(json_path, "r") as f:
        all_patterns = json.load(f)
    return all_patterns["fold_1"]


def build_feature_names(patterns, series_names):
    """Create display names for each pattern feature."""
    feature_names = []
    for i, pattern in enumerate(patterns):
        series_idx = pattern["channel"]
        transform = pattern["transform"]
        if series_idx < len(series_names):
            series_name = series_names[series_idx]
        else:
            series_name = f"channel_{series_idx}"
        feature_names.append(f"P{i + 1}: {series_name} ({transform})")
    return feature_names


def extract_features(X_list, patterns):
    """Extract features using PatternExtractor utilities."""
    extractor = PatternExtractor(
        model=LightGBMWrapper(task_type="classification", n_classes=2, inner_cv=1),
        metric="accuracy",
        n_trials=1,
        n_trials_without_improvement=1,
        show_progress=False,
        verbose=False,
    )
    return extractor.extract_features(X_list, patterns=patterns)


def select_top_shap_features(model, X, feature_names, top_k=6, seed=42):
    """Select top features by mean absolute SHAP value."""
    rng = np.random.default_rng(seed)
    if X.shape[0] > 1000:
        idx = rng.choice(X.shape[0], size=1000, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_sub)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_k]

    top_names = [feature_names[i] for i in top_idx]
    return top_idx, top_names, mean_abs


def plot_decision_tree(
    tree_model, feature_names, output_path, title, class_names
):
    """Plot a clinician-friendly decision tree with class labels only at leaves."""
    from sklearn.tree import export_graphviz
    import re
    
    # Export to dot format
    # Use filled=True so graphviz respects fillcolor attributes
    dot_data = export_graphviz(
        tree_model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,  # Needed for fillcolor to work
        rounded=True,
        proportion=True,
        impurity=False,
        special_characters=True,
    )
    
    # Modify: remove class labels from internal nodes, keep only at leaves
    # Also remove value arrays from all nodes
    lines = dot_data.split('\n')
    modified_lines = []
    
    for line in lines:
        if 'label=' in line and '[' in line:
            # Extract node ID from the line (first number before [label=)
            match = re.search(r'(\d+)\s+\[label=', line)
            if match:
                node_id = int(match.group(1))
                is_leaf = tree_model.tree_.children_left[node_id] == -1
                
                # Remove value array from all nodes (HTML format)
                line = re.sub(r'<br/>value = \[[^\]]*\]', '', line)
                # Remove value array (regular format)
                line = re.sub(r'\\nvalue = \[[^\]]*\]', '', line)
                line = re.sub(r'\\\\nvalue = \[[^\]]*\]', '', line)
                
                if not is_leaf:
                    # Remove class label from internal nodes
                    # Handle both formats:
                    # 1. Escaped newline format: ...\\nclass = ClassName"
                    # 2. HTML format (with special_characters=True): ...<br/>class = ClassName>
                    # 3. Plain format: ...class = ClassName"
                    
                    # HTML format (most common with special_characters=True)
                    line = re.sub(r'<br/>class = [^>]*', '', line)
                    # Escaped newline format
                    line = re.sub(r'\\\\nclass = [^"]*', '', line)
                    line = re.sub(r'\\nclass = [^"]*', '', line)
                    # Plain format (comma-separated)
                    line = re.sub(r',\s*class = [^"]*', '', line)
                    # Clean up any trailing issues
                    line = line.replace('", fillcolor', '", fillcolor')
                    line = line.replace('><br/>', '>')  # Clean up leftover <br/> tags
        
        modified_lines.append(line)
    
    # Add consistent styling - light blue for internal nodes, light green for leaves
    styled_lines = []
    for line in modified_lines:
        if '[label=' in line and not line.strip().startswith('//'):
            # Check if it's a leaf node and add appropriate color
            match = re.search(r'(\d+)\s+\[label=', line)
            if match:
                node_id = int(match.group(1))
                is_leaf = tree_model.tree_.children_left[node_id] == -1
                # Remove existing fillcolor and style attributes
                line = re.sub(r',\s*fillcolor="[^"]*"', '', line)
                line = re.sub(r',\s*style="[^"]*"', '', line)
                
                # Determine color based on node type and predicted class
                if is_leaf:
                    # Get the predicted class for this leaf node
                    # The value array shows [class_0_count, class_1_count]
                    node_value = tree_model.tree_.value[node_id][0]
                    predicted_class = int(node_value.argmax())
                    # Green for "No ARDS" (class 0), Red for "ARDS" (class 1)
                    if predicted_class == 0:
                        color = "#90EE90"  # Light green for No ARDS
                    else:
                        color = "#FFB6C1"  # Light red/pink for ARDS
                else:
                    color = "#ADD8E6"  # Light blue for internal nodes
                
                # Handle both HTML format (label=<...>) and regular format (label="...")
                # Need to add style="filled" for graphviz to respect fillcolor
                if 'label=<' in line:
                    # HTML format: ...>] ; -> ...>, style="filled", fillcolor="color"] ;
                    line = re.sub(r'>\]\s*;', f'>, style="filled", fillcolor="{color}"] ;', line)
                else:
                    # Regular format: ..."] ; -> ..., style="filled", fillcolor="color"] ;
                    line = re.sub(r'"\]\s*;', f'", style="filled", fillcolor="{color}"] ;', line)
        styled_lines.append(line)
    
    styled_dot = '\n'.join(styled_lines)
    
    # Try to use graphviz if available, otherwise use matplotlib
    try:
        import graphviz
        graph = graphviz.Source(styled_dot)
        png_path = output_path.replace('.png', '')
        graph.render(png_path, format='png', cleanup=True)
        import os
        if os.path.exists(png_path + '.png'):
            os.rename(png_path + '.png', output_path)
            print(f"  Used graphviz to render tree (class labels only at leaves)")
    except (ImportError, Exception) as e:
        # Fallback: use matplotlib (will still show class at all nodes, but better than nothing)
        print(f"  Graphviz not available, using matplotlib fallback: {e}")
        fig, ax = plt.subplots(figsize=(20, 12))
        plot_tree(
            tree_model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            proportion=True,
            impurity=False,
            fontsize=9,
            ax=ax,
        )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()


def main():
    """Main function to create SHAP-guided decision tree."""
    print("Loading MIMIC data...")
    X_list, y, series_names = load_mimic_data()

    print("Loading patterns from fold_1...")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    json_path = os.path.join(
        base_dir, "json_files", "mimic", "pattern_parameters.json"
    )
    patterns = load_patterns_from_json(json_path)
    print(f"Loaded {len(patterns)} patterns")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = list(skf.split(pd.concat(X_list, axis=1), y))[0]

    print("Extracting features...")
    X_train_feat = extract_features([x.iloc[train_idx] for x in X_list], patterns)
    X_test_feat = extract_features([x.iloc[test_idx] for x in X_list], patterns)
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print(f"Training features shape: {X_train_feat.shape}")
    print(f"Test features shape: {X_test_feat.shape}")

    feature_names = build_feature_names(patterns, series_names)

    print("Training LightGBM model...")
    lgb_model = LightGBMWrapper(task_type="classification", n_classes=2, inner_cv=1)
    lgb_model.fit(X_train_feat, y_train.values)
    lgb_preds = lgb_model.predict(X_test_feat)
    lgb_acc = accuracy_score(y_test.values, lgb_preds)
    print(f"LightGBM - Test Accuracy: {lgb_acc:.4f}")

    print("Selecting top SHAP features...")
    top_idx, top_names, _ = select_top_shap_features(
        lgb_model, X_train_feat, feature_names, top_k=6
    )

    print("Training decision tree (max_depth=4)...")
    tree_model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=25,
        random_state=42,
    )
    tree_model.fit(X_train_feat[:, top_idx], y_train.values)
    tree_preds = tree_model.predict(X_test_feat[:, top_idx])
    tree_acc = accuracy_score(y_test.values, tree_preds)
    print(f"Decision Tree - Test Accuracy: {tree_acc:.4f}")

    output_path = os.path.join(
        base_dir, "elsarticle", "images", "mimic_shap_tree.png"
    )
    print("Saving decision tree plot...")
    plot_decision_tree(
        tree_model,
        top_names,
        output_path,
        title="ARDS Decision Tree (SHAP-guided patX patterns)",
        class_names=["No ARDS", "ARDS"],
    )
    print(f"Saved decision tree to {output_path}")


if __name__ == "__main__":
    main()
