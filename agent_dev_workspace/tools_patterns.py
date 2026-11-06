import os
import tempfile
from typing import Literal, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> dict:
        """Mine frequent patterns in a specific category using PAMI algorithms.

        Filters session.dataframe to category_name, builds a category-local DTM with
        CountVectorizer(stop_words='english'), applies percentile filtering, converts to
        a transactional database, and runs the requested PAMI algorithm.

        Args:
            category_name: Name of category to mine (matches category_name or category column).
            filtering_method: Method to filter terms - 'variance' (5th-95th percentile),
                              'tfidf' (>= 20th percentile), or 'term_frequency' (5th-95th percentile).
            algorithm: Mining algorithm - 'fpgrowth', 'topk', or 'maxfpgrowth' (default: 'fpgrowth').
            min_sup: Minimum support threshold (int). Required for 'fpgrowth' and 'maxfpgrowth'.
            k: Top-k pattern count (int). Required for 'topk'.

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm,
            pattern_count, patterns list, and optional note.
        """
        df = session.dataframe
        if df is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        # Validate required algorithm parameters
        algo_lower = algorithm.lower()
        if algo_lower in ("fpgrowth", "maxfpgrowth") and (min_sup is None or min_sup <= 0):
            return {"error": f"Algorithm '{algorithm}' requires a positive integer 'min_sup' parameter."}

        if algo_lower == "topk" and (k is None or k <= 0):
            return {"error": "Algorithm 'topk' requires a positive integer 'k' parameter."}

        # Filter dataframe by category_name (checking category_name or category columns)
        if "category_name" in df.columns:
            cat_mask = df["category_name"] == category_name
        elif "category" in df.columns:
            cat_mask = df["category"] == category_name
        else:
            return {"error": "No category or category_name column found in dataset."}

        cat_df = df[cat_mask]
        if len(cat_df) == 0:
            return {"error": f"Category '{category_name}' not found or has 0 rows in dataset."}

        # 1. Build category-local DTM with stop_words='english'
        vectorizer = CountVectorizer(stop_words="english")
        texts = cat_df["text"].fillna("").astype(str)
        try:
            X_cat = vectorizer.fit_transform(texts)
        except ValueError:
            # Raised if vocabulary is empty after stopword removal
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "pattern_count": 0,
                "note": "no terms survived stopword removal",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        feature_names = np.array(vectorizer.get_feature_names_out())
        if len(feature_names) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "pattern_count": 0,
                "note": "no terms survived stopword removal",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # 2. Apply chosen filtering_method
        method_lower = filtering_method.lower()
        if method_lower == "variance":
            # Keep middle 90% (5th to 95th percentile)
            variances = np.asarray(X_cat.toarray().var(axis=0)).ravel()
            p5, p95 = np.percentile(variances, [5, 95])
            kept_mask = (variances > p5) & (variances < p95)

        elif method_lower == "term_frequency":
            # Keep middle 90% (5th to 95th percentile)
            frequencies = np.asarray(X_cat.sum(axis=0)).ravel()
            p5, p95 = np.percentile(frequencies, [5, 95])
            kept_mask = (frequencies > p5) & (frequencies < p95)

        elif method_lower == "tfidf":
            # Keep top 80% (>= 20th percentile of mean TF-IDF)
            tfidf_mat = TfidfTransformer().fit_transform(X_cat)
            mean_tfidf = np.asarray(tfidf_mat.mean(axis=0)).ravel()
            p20 = np.percentile(mean_tfidf, 20)
            kept_mask = mean_tfidf >= p20

        else:
            return {"error": f"Unsupported filtering_method '{filtering_method}'."}

        kept_terms = feature_names[kept_mask]

        # 3. Handle zero terms kept case cleanly
        if len(kept_terms) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "pattern_count": 0,
                "note": "no terms survived filtering",
            }
            session.store_result(
                "mine_patterns_tool",
                {
                    "category_name": category_name,
                    "filtering_method": filtering_method,
                    "algorithm": algorithm,
                },
                summary,
            )
            artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
            session.artifacts[artifact_key] = []
            return summary

        # 4. Convert kept terms DTM to transactional database & run PAMI
        X_kept = X_cat[:, kept_mask].toarray()
        kept_df = pd.DataFrame(X_kept, columns=kept_terms)

        patterns_list = []

        with tempfile.TemporaryDirectory() as temp_dir:
            db_file = os.path.join(temp_dir, "transactional_db.txt")
            out_file = os.path.join(temp_dir, "patterns_out.txt")

            try:
                from PAMI.extras.convert import DF2DB
                # Match Master section 5.4.2: DF2DB(kept_df) and convert2TransactionalDatabase(db_file, '>=', 1)
                converter = DF2DB.DF2DB(kept_df)
                converter.convert2TransactionalDatabase(db_file, ">=", 1)
            except Exception as e:
                return {"error": f"Failed to convert DataFrame to PAMI transactional database: {str(e)}"}

            # Run requested PAMI algorithm using .mine() (matching Master notebook)
            try:
                if algo_lower == "fpgrowth":
                    from PAMI.frequentPattern.basic import FPGrowth
                    miner = FPGrowth.FPGrowth(iFile=db_file, minSup=min_sup, sep="\t")
                    miner.mine()
                    miner.save(out_file)

                elif algo_lower == "topk":
                    from PAMI.frequentPattern.topk import FAE
                    miner = FAE.FAE(iFile=db_file, k=k, sep="\t")
                    miner.mine()
                    miner.save(out_file)

                elif algo_lower == "maxfpgrowth":
                    from PAMI.frequentPattern.maximal import MaxFPGrowth
                    miner = MaxFPGrowth.MaxFPGrowth(iFile=db_file, minSup=min_sup, sep="\t")
                    miner.mine()
                    miner.save(out_file)

                else:
                    return {"error": f"Unsupported algorithm '{algorithm}'."}

                # Read mined patterns dataframe if output file exists and is non-empty
                if hasattr(miner, "getPatternsAsDataFrame"):
                    pat_df = miner.getPatternsAsDataFrame()
                    if pat_df is not None and len(pat_df) > 0:
                        for _, row in pat_df.iterrows():
                            # Format pattern string and support count
                            pat_str = str(row.iloc[0])
                            sup_val = int(row.iloc[1])
                            patterns_list.append({"pattern": pat_str, "support": sup_val})
                elif os.path.exists(out_file) and os.path.getsize(out_file) > 0:
                    pat_df = pd.read_csv(out_file, sep="\t", header=None, names=["pattern", "support"])
                    for _, row in pat_df.iterrows():
                        patterns_list.append({"pattern": str(row["pattern"]), "support": int(row["support"])})

            except Exception as e:
                return {"error": f"PAMI pattern mining failed: {str(e)}"}

        # Store artifacts and result summary
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns_list

        result_id = session.next_result_id("mine_patterns")

        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "pattern_count": len(patterns_list),
            "patterns": patterns_list,
        }

        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k,
            },
            summary,
        )

        return summary

    return [mine_patterns_tool]
