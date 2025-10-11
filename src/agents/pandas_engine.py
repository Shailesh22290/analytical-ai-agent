"""
Deterministic pandas analysis engine
All numeric computations happen here - no LLM involvement
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

from src.agents.ingestion import csv_ingestion
from src.utils.models import (
    CompareAveragesParams,
    FilterThresholdParams,
    SortParams,
    TopNParams,
    CompareTopParams
)


class PandasEngine:
    """Executes deterministic pandas operations"""
    
    def compare_averages(
        self, 
        params: CompareAveragesParams
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Compare averages of a column across files or groups
        
        Args:
            params: CompareAveragesParams
            
        Returns:
            (result_table, numbers)
        """
        column = params.column
        
        if params.file1_id and params.file2_id:
            # Compare across two files
            df1 = csv_ingestion.get_dataframe(params.file1_id)
            df2 = csv_ingestion.get_dataframe(params.file2_id)
            
            if column not in df1.columns or column not in df2.columns:
                raise ValueError(f"Column {column} not found in both files")
            
            avg1 = float(df1[column].mean())
            avg2 = float(df2[column].mean())
            diff = avg1 - avg2
            pct_diff = (diff / avg2 * 100) if avg2 != 0 else 0
            
            numbers = {
                f"{params.file1_id}_average": avg1,
                f"{params.file2_id}_average": avg2,
                "difference": diff,
                "percent_difference": pct_diff
            }
            
            result_table = [
                {"file_id": params.file1_id, "average": avg1},
                {"file_id": params.file2_id, "average": avg2}
            ]
            
        elif params.group_by:
            # Group by and compare within single file
            file_id = params.file1_id or list(csv_ingestion.dataframes.keys())[0]
            df = csv_ingestion.get_dataframe(file_id)
            
            if column not in df.columns or params.group_by not in df.columns:
                raise ValueError(f"Columns not found")
            
            grouped = df.groupby(params.group_by)[column].mean()
            
            result_table = [
                {params.group_by: str(group), "average": float(avg)}
                for group, avg in grouped.items()
            ]
            
            numbers = {
                f"average_by_{params.group_by}": {
                    str(group): float(avg) for group, avg in grouped.items()
                },
                "overall_average": float(df[column].mean()),
                "max_average": float(grouped.max()),
                "min_average": float(grouped.min())
            }
            
        else:
            # Single file average
            file_id = params.file1_id or list(csv_ingestion.dataframes.keys())[0]
            df = csv_ingestion.get_dataframe(file_id)
            
            if column not in df.columns:
                raise ValueError(f"Column {column} not found")
            
            avg = float(df[column].mean())
            std = float(df[column].std())
            min_val = float(df[column].min())
            max_val = float(df[column].max())
            
            numbers = {
                "average": avg,
                "std_dev": std,
                "min": min_val,
                "max": max_val,
                "count": len(df)
            }
            
            result_table = [{"metric": k, "value": v} for k, v in numbers.items()]
        
        return result_table, numbers
    
    def filter_threshold(
        self, 
        params: FilterThresholdParams
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Filter rows based on threshold
        
        Args:
            params: FilterThresholdParams
            
        Returns:
            (result_table, numbers)
        """
        file_id = params.file_id or list(csv_ingestion.dataframes.keys())[0]
        df = csv_ingestion.get_dataframe(file_id)
        
        if params.column not in df.columns:
            raise ValueError(f"Column {params.column} not found")
        
        # Apply filter
        if params.operator == '>':
            filtered_df = df[df[params.column] > params.value]
        elif params.operator == '<':
            filtered_df = df[df[params.column] < params.value]
        elif params.operator == '>=':
            filtered_df = df[df[params.column] >= params.value]
        elif params.operator == '<=':
            filtered_df = df[df[params.column] <= params.value]
        elif params.operator == '==':
            filtered_df = df[df[params.column] == params.value]
        elif params.operator == '!=':
            filtered_df = df[df[params.column] != params.value]
        else:
            raise ValueError(f"Invalid operator: {params.operator}")
        
        # Get results
        result_table = filtered_df.head(100).to_dict('records')
        
        numbers = {
            "total_rows": len(df),
            "filtered_rows": len(filtered_df),
            "filter_column": params.column,
            "filter_operator": params.operator,
            "filter_value": params.value,
            "percentage_matched": (len(filtered_df) / len(df) * 100) if len(df) > 0 else 0,
            "row_indices": filtered_df.index.tolist()[:100]
        }
        
        return result_table, numbers
    
    def sort_data(
        self, 
        params: SortParams
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Sort data by column
        
        Args:
            params: SortParams
            
        Returns:
            (result_table, numbers)
        """
        file_id = params.file_id or list(csv_ingestion.dataframes.keys())[0]
        df = csv_ingestion.get_dataframe(file_id)
        
        if params.column not in df.columns:
            raise ValueError(f"Column {params.column} not found")
        
        sorted_df = df.sort_values(by=params.column, ascending=params.ascending)
        
        if params.limit:
            sorted_df = sorted_df.head(params.limit)
        
        result_table = sorted_df.to_dict('records')
        
        numbers = {
            "total_rows": len(df),
            "returned_rows": len(sorted_df),
            "sort_column": params.column,
            "ascending": params.ascending,
            "first_value": sorted_df[params.column].iloc[0] if len(sorted_df) > 0 else None,
            "last_value": sorted_df[params.column].iloc[-1] if len(sorted_df) > 0 else None,
            "row_indices": sorted_df.index.tolist()
        }
        
        return result_table, numbers
    
    def top_n(
        self, 
        params: TopNParams
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Get top N rows by column value
        
        Args:
            params: TopNParams
            
        Returns:
            (result_table, numbers)
        """
        file_id = params.file_id or list(csv_ingestion.dataframes.keys())[0]
        df = csv_ingestion.get_dataframe(file_id)
        
        if params.column not in df.columns:
            raise ValueError(f"Column {params.column} not found")
        
        sorted_df = df.sort_values(by=params.column, ascending=params.ascending)
        top_df = sorted_df.head(params.n)
        
        result_table = top_df.to_dict('records')
        
        # Extract actual values
        top_values = top_df[params.column].tolist()
        
        numbers = {
            "n": params.n,
            "column": params.column,
            "top_values": top_values,
            "top_indices": top_df.index.tolist(),
            "highest_value": top_values[0] if top_values else None,
            "lowest_in_top": top_values[-1] if top_values else None,
            "average_of_top": float(np.mean(top_values)) if top_values else None
        }
        
        return result_table, numbers
    
    def compare_top(
        self, 
        params: CompareTopParams
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Compare top N items across two files
        
        Args:
            params: CompareTopParams
            
        Returns:
            (result_table, numbers)
        """
        if not params.file1_id or not params.file2_id:
            raise ValueError("Both file1_id and file2_id required for compare_top")
        
        df1 = csv_ingestion.get_dataframe(params.file1_id)
        df2 = csv_ingestion.get_dataframe(params.file2_id)
        
        if params.column not in df1.columns or params.column not in df2.columns:
            raise ValueError(f"Column {params.column} not found in both files")
        
        # Get top N from each
        top1 = df1.nlargest(params.n, params.column)
        top2 = df2.nlargest(params.n, params.column)
        
        # Combine for result table
        result_table = []
        for i in range(params.n):
            row = {"rank": i + 1}
            if i < len(top1):
                row[f"{params.file1_id}_{params.column}"] = top1.iloc[i][params.column]
                row[f"{params.file1_id}_index"] = int(top1.index[i])
            if i < len(top2):
                row[f"{params.file2_id}_{params.column}"] = top2.iloc[i][params.column]
                row[f"{params.file2_id}_index"] = int(top2.index[i])
            result_table.append(row)
        
        numbers = {
            "n": params.n,
            "column": params.column,
            f"{params.file1_id}_top_values": top1[params.column].tolist(),
            f"{params.file2_id}_top_values": top2[params.column].tolist(),
            f"{params.file1_id}_average": float(top1[params.column].mean()),
            f"{params.file2_id}_average": float(top2[params.column].mean()),
            f"{params.file1_id}_max": float(top1[params.column].max()),
            f"{params.file2_id}_max": float(top2[params.column].max())
        }
        
        return result_table, numbers


# Global engine instance
pandas_engine = PandasEngine()