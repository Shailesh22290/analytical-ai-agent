"""
Unit tests for the Analytical Agent
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.agents.ingestion import csv_ingestion
from src.agents.pandas_engine import pandas_engine
from src.agents.analytical_agent import analytical_agent
from src.utils.models import (
    CompareAveragesParams,
    FilterThresholdParams,
    TopNParams
)


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a sample CSV file for testing"""
    data = {
        'id': range(1, 11),
        'name': [f'Product_{i}' for i in range(1, 11)],
        'price': [10.5, 20.0, 15.5, 30.0, 25.5, 12.0, 18.5, 22.0, 16.5, 28.0],
        'quantity': [100, 50, 75, 30, 45, 90, 60, 40, 80, 35],
        'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C']
    }
    df = pd.DataFrame(data)
    
    csv_path = tmp_path / "test_products.csv"
    df.to_csv(csv_path, index=False)
    
    return str(csv_path)


@pytest.fixture
def sample_csv_file2(tmp_path):
    """Create a second sample CSV file for comparison tests"""
    data = {
        'id': range(1, 11),
        'name': [f'Item_{i}' for i in range(1, 11)],
        'price': [12.0, 22.0, 17.0, 32.0, 27.0, 14.0, 20.0, 24.0, 18.0, 30.0],
        'quantity': [110, 55, 80, 35, 50, 95, 65, 45, 85, 40]
    }
    df = pd.DataFrame(data)
    
    csv_path = tmp_path / "test_items.csv"
    df.to_csv(csv_path, index=False)
    
    return str(csv_path)


class TestIngestion:
    """Test CSV ingestion"""
    
    def test_ingest_csv_basic(self, sample_csv_file):
        """Test basic CSV ingestion"""
        file_id, metadata = csv_ingestion.ingest_csv(
            sample_csv_file,
            file_id="test1",
            vectorize=False
        )
        
        assert file_id == "test1"
        assert metadata.num_rows == 10
        assert metadata.num_columns == 5
        assert 'price' in metadata.numeric_columns
        assert 'name' in metadata.text_columns
    
    def test_dataframe_retrieval(self, sample_csv_file):
        """Test dataframe retrieval"""
        file_id, _ = csv_ingestion.ingest_csv(
            sample_csv_file,
            file_id="test2",
            vectorize=False
        )
        
        df = csv_ingestion.get_dataframe(file_id)
        assert len(df) == 10
        assert 'price' in df.columns


class TestPandasEngine:
    """Test pandas analysis engine"""
    
    def test_filter_threshold(self, sample_csv_file):
        """Test threshold filtering"""
        file_id, _ = csv_ingestion.ingest_csv(
            sample_csv_file,
            file_id="test_filter",
            vectorize=False
        )
        
        params = FilterThresholdParams(
            column="price",
            operator=">",
            value=20.0,
            file_id=file_id
        )
        
        result_table, numbers = pandas_engine.filter_threshold(params)
        
        assert numbers['filtered_rows'] == 4  # Prices > 20: 20.0 is not included
        assert all(row['price'] > 20.0 for row in result_table)
    
    def test_top_n(self, sample_csv_file):
        """Test top N retrieval"""
        file_id, _ = csv_ingestion.ingest_csv(
            sample_csv_file,
            file_id="test_topn",
            vectorize=False
        )
        
        params = TopNParams(
            column="price",
            n=3,
            ascending=False,
            file_id=file_id
        )
        
        result_table, numbers = pandas_engine.top_n(params)
        
        assert len(result_table) == 3
        assert numbers['highest_value'] == 30.0
        assert numbers['n'] == 3
    
    def test_compare_averages_single_file(self, sample_csv_file):
        """Test average calculation"""
        file_id, _ = csv_ingestion.ingest_csv(
            sample_csv_file,
            file_id="test_avg",
            vectorize=False
        )
        
        params = CompareAveragesParams(
            column="price",
            file1_id=file_id
        )
        
        result_table, numbers = pandas_engine.compare_averages(params)
        
        assert 'average' in numbers
        assert numbers['average'] == pytest.approx(19.85, rel=0.01)
    
    def test_compare_averages_two_files(self, sample_csv_file, sample_csv_file2):
        """Test comparing averages across two files"""
        file_id1, _ = csv_ingestion.ingest_csv(
            sample_csv_file,
            file_id="test_cmp1",
            vectorize=False
        )
        
        file_id2, _ = csv_ingestion.ingest_csv(
            sample_csv_file2,
            file_id="test_cmp2",
            vectorize=False
        )
        
        params = CompareAveragesParams(
            column="price",
            file1_id=file_id1,
            file2_id=file_id2
        )
        
        result_table, numbers = pandas_engine.compare_averages(params)
        
        assert 'test_cmp1_average' in numbers
        assert 'test_cmp2_average' in numbers
        assert 'difference' in numbers
        assert len(result_table) == 2


class TestAnalyticalAgent:
    """Test the main analytical agent"""
    
    def test_agent_status_no_data(self):
        """Test agent status with no data loaded"""
        status = analytical_agent.get_status()
        
        assert status['status'] == 'ready'
        assert isinstance(status['supported_intents'], list)
    
    def test_unsupported_intent(self, sample_csv_file):
        """Test handling of unsupported intent"""
        csv_ingestion.ingest_csv(
            sample_csv_file,
            file_id="test_unsupported",
            vectorize=False
        )
        
        # Mock an unsupported intent by directly calling
        result = analytical_agent.process_query("do something impossible")
        
        # Should handle gracefully
        assert 'error' in result or 'narrative' in result


def test_integration_workflow(sample_csv_file, sample_csv_file2):
    """Test complete workflow"""
    # Ingest files
    file_id1, _ = csv_ingestion.ingest_csv(
        sample_csv_file,
        file_id="integration1",
        vectorize=False
    )
    
    file_id2, _ = csv_ingestion.ingest_csv(
        sample_csv_file2,
        file_id="integration2",
        vectorize=False
    )
    
    # Check status
    status = analytical_agent.get_status()
    assert status['loaded_files'] >= 2
    
    # Test various queries through pandas engine
    # (Note: Full LLM integration would require API key and network)
    
    # Filter test
    filter_params = FilterThresholdParams(
        column="price",
        operator=">=",
        value=25.0,
        file_id=file_id1
    )
    result_table, numbers = pandas_engine.filter_threshold(filter_params)
    assert numbers['filtered_rows'] > 0
    
    # Top N test
    topn_params = TopNParams(
        column="quantity",
        n=5,
        file_id=file_id1
    )
    result_table, numbers = pandas_engine.top_n(topn_params)
    assert len(result_table) == 5
    
    print("\n✅ Integration test passed!")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])