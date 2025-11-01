"""
Enhanced Dashboard Wrapper for AI Agent
32-Question Structure: 8 Observations + 8 Analysis + Motor/Pump Assessments
"""
import sys
from pathlib import Path
import logging
import json
import time  # <-- [FIX] Import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedDashboardAnalysisAPI:
    """Enhanced API for dashboard to call AI Agent with 32-question structure"""
    
    def __init__(self):
        """Initialize the AI Agent components"""
        try:
            from src.agents.document_ingestion import DocumentIngestion
            from src.agents.analytical_agent import AnalyticalAgent
            from config.settings import settings
            
            self.document_ingestion = DocumentIngestion()
            self.analytical_agent = AnalyticalAgent()
            self.settings = settings
            
            logger.info("✅ EnhancedDashboardAnalysisAPI initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import AI Agent modules: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize EnhancedDashboardAnalysisAPI: {e}")
            raise

    def _get_observation_prompts(self):
        """Return 8 observation prompts - extract numeric data and comparisons"""
        return [
            {
                'id': 'Q1_OBS',
                'category': 'OBSERVATIONS',
                'parameter': 'A',
                'text': 'From the Parameter Assessment table, extract: How much has the Overall envelope value (Parameter A) increased compared to its baseline? Provide: current value, baseline value, and percentage rise.',
                'expected_format': 'Overall envelope value has increased to [X value in gE] (current value) compared to [Y value in gE] (baseline). This rise is around [Z%].',
                'data_type': 'numeric'
            },
            {
                'id': 'Q2_OBS',
                'category': 'OBSERVATIONS',
                'parameter': 'B',
                'text': 'From the Parameter Assessment table, extract: How much has the Overall acceleration value (Parameter B) increased compared to its baseline? Provide: current value, baseline value, and percentage rise.',
                'expected_format': 'Overall acceleration value has increased to [X value in mm/s²] (current value) compared to [Y value in mm/s²] (baseline). This rise is around [Z%].',
                'data_type': 'numeric'
            },
            {
                'id': 'Q3_OBS',
                'category': 'OBSERVATIONS',
                'parameter': 'G',
                'text': 'From the Parameter Assessment table, extract: How much has the Total harmonic energy (Parameter G) increased compared to its baseline? Provide: current value, baseline value, and percentage rise.',
                'expected_format': 'Total harmonic energy value has increased to [X value] (current value) compared to [Y value] (baseline). This rise is around [Z%].',
                'data_type': 'numeric'
            },
            {
                'id': 'Q4_OBS',
                'category': 'OBSERVATIONS',
                'parameter': 'C-F',
                'text': 'From the Parameter Assessment table, compare BPFO (C), BPFI (D), BSF (E), and FTF (F) harmonic energies. Which fault frequency shows the HIGHEST percentage increase from baseline? List all four with their rise percentages.',
                'expected_format': 'Among fault frequencies: BPFO increased [Z1%], BPFI increased [Z2%], BSF increased [Z3%], FTF increased [Z4%]. The highest increase is [FAULT_TYPE] at [Z_max%].',
                'data_type': 'comparative'
            },
            {
                'id': 'Q5_OBS',
                'category': 'OBSERVATIONS',
                'parameter': 'H',
                'text': 'From the Parameter Assessment table, extract: How much has the Running speed sideband energy around BPFI harmonics (Parameter H) increased compared to its baseline? Provide: current value, baseline value, and percentage rise.',
                'expected_format': 'Running speed sideband energy around BPFI has increased to [X value] (current value) compared to [Y value] (baseline). This rise is around [Z%].',
                'data_type': 'numeric'
            },
            {
                'id': 'Q6_OBS',
                'category': 'OBSERVATIONS',
                'parameter': 'I',
                'text': 'From the Parameter Assessment table, extract: How much has the Kurtosis value (Parameter I) in the acceleration waveform increased compared to its baseline? Provide: current value, baseline value, and percentage rise.',
                'expected_format': 'Kurtosis value has increased to [X value] (current value) compared to [Y value] (baseline). This rise is around [Z%].',
                'data_type': 'numeric'
            },
            {
                'id': 'Q7_OBS',
                'category': 'OBSERVATIONS',
                'parameter': 'J',
                'text': 'From the Parameter Assessment table, extract: How much has the Crest Factor value (Parameter J) in the acceleration waveform increased compared to its baseline? Provide: current value, baseline value, and percentage rise.',
                'expected_format': 'Crest factor value has increased to [X value] (current value) compared to [Y value] (baseline). This rise is around [Z%].',
                'data_type': 'numeric'
            },
            {
                'id': 'Q8_OBS',
                'category': 'OBSERVATIONS',
                'parameter': 'Risk',
                'text': 'From the Assessment Summary table, what is the Overall Score, Risk Level, and Condition Stage reported? Extract exact values.',
                'expected_format': 'The overall score is [X], the risk level is [RISK_LEVEL], and the condition stage is [STAGE].',
                'data_type': 'summary'
            }
        ]

    def _get_analysis_prompts(self):
        """Return 8 analysis prompts - interpret observations and identify faults"""
        return [
            {
                'id': 'Q1_ANALYSIS',
                'category': 'ANALYSIS',
                'parameter': 'A',
                'text': 'Based on the Overall envelope value (Parameter A) observation: What does this [Z%] increase in envelope value indicate about bearing condition? Does it suggest early stage degradation, accelerated wear, or critical failure imminent?',
                'expected_format': 'The [Z%] rise in overall envelope value indicates [bearing condition assessment]. This suggests [stage of degradation]. This is a [severity level] indicator of bearing health deterioration.',
                'data_type': 'interpretation'
            },
            {
                'id': 'Q2_ANALYSIS',
                'category': 'ANALYSIS',
                'parameter': 'B',
                'text': 'Based on the Overall acceleration increase (Parameter B) observation: What does this [Z%] increase in absolute acceleration suggest about the intensity of vibration and bearing mechanical state? Is this consistent with lubrication issues or bearing damage?',
                'expected_format': 'The [Z%] rise in overall acceleration indicates [intensity of vibration]. This suggests [mechanical state]. The pattern is consistent with [lubrication issues/bearing damage/other].',
                'data_type': 'interpretation'
            },
            {
                'id': 'Q3_ANALYSIS',
                'category': 'ANALYSIS',
                'parameter': 'G',
                'text': 'Based on Total harmonic energy increase (Parameter G) observation: What does this [Z%] rise in total harmonic energy indicate? Does it show presence of multiple bearing fault frequencies or dominant single fault development?',
                'expected_format': 'The [Z%] rise in total harmonic energy indicates [fault pattern]. This suggests [single/multiple] bearing fault frequency dominance. The increase pattern is characteristic of [bearing degradation stage].',
                'data_type': 'interpretation'
            },
            {
                'id': 'Q4_ANALYSIS',
                'category': 'ANALYSIS',
                'parameter': 'C-F',
                'text': 'Based on the fault frequency comparison: The [FAULT_TYPE] shows the highest increase at [Z_max%]. What specific bearing component is damaged? What is the failure mechanism (spalling, cracking, contamination, lubrication starvation)? How does this compare to BSF and FTF signals?',
                'expected_format': '[FAULT_TYPE] dominance at [Z_max%] indicates damage to [bearing component: outer race/inner race/rolling elements/cage]. The failure mechanism appears to be [mechanism]. Other fault frequencies [comparison statement]. This is characteristic of [fault stage].',
                'data_type': 'interpretation'
            },
            {
                'id': 'Q5_ANALYSIS',
                'category': 'ANALYSIS',
                'parameter': 'H',
                'text': 'Based on Running speed sideband energy increase (Parameter H): What does this [Z%] rise in sidebands around BPFI harmonics indicate about raceway damage? Does sideband growth suggest modulation and progressive bearing failure?',
                'expected_format': 'The [Z%] rise in running speed sidebands around BPFI indicates [raceway damage assessment]. Sideband modulation pattern suggests [progressive failure stage]. This is characteristic of [bearing fault type: inner race/outer race/rolling element].',
                'data_type': 'interpretation'
            },
            {
                'id': 'Q6_ANALYSIS',
                'category': 'ANALYSIS',
                'parameter': 'I',
                'text': 'Based on Kurtosis increase (Parameter I): The [Z%] rise in kurtosis from [Y] to [X] indicates what about impacting and irregularities? Does elevated kurtosis combined with other parameters confirm bearing spalling or raceway damage?',
                'expected_format': 'The [Z%] increase in kurtosis to [X] value indicates [level of impacting]. This confirms [raceway irregularities/spalling/damage type]. Elevated kurtosis combined with [other parameters] strongly indicates [bearing fault confirmation].',
                'data_type': 'interpretation'
            },
            {
                'id': 'Q7_ANALYSIS',
                'category': 'ANALYSIS',
                'parameter': 'J',
                'text': 'Based on Crest Factor increase (Parameter J): The [Z%] rise in crest factor to [X] suggests what about signal peakiness and shock events? Does this indicate repetitive impacting and bearing raceway surface degradation?',
                'expected_format': 'The [Z%] increase in crest factor to [X] value indicates [signal peakiness level]. This suggests [shock event intensity]. Repetitive impacting pattern indicates [raceway surface degradation stage]. Combined with kurtosis, this confirms [bearing damage progression].',
                'data_type': 'interpretation'
            },
            {
                'id': 'Q8_ANALYSIS',
                'category': 'ANALYSIS',
                'parameter': 'Risk',
                'text': 'Based on the overall condition stage and risk level: If the bearing is in [STAGE] with [RISK_LEVEL] risk, what is the recommended maintenance action timeline? Should action be immediate, urgent (1-2 weeks), or routine follow-up?',
                'expected_format': 'At [STAGE] with [RISK_LEVEL] risk, the bearing requires [maintenance action]. Timeline: [immediate/1-2 weeks/4 weeks/routine follow-up]. This is justified because [risk justification].',
                'data_type': 'recommendation_basis'
            }
        ]

    def _get_equipment_assessment_prompts(self):
        """Return motor and pump specific assessment prompts (16 total)"""
        return {
            'motor': [
                {
                    'id': 'MOTOR_Q1',
                    'equipment': 'Motor',
                    'parameter': 'MND',
                    'text': 'For the MOTOR - Non-Drive End (MND) bearing: Considering all observation and analysis findings, summarize the current bearing health status. Include dominant fault type, severity, and risk of motor winding contamination from bearing particles.',
                    'expected_format': 'Motor MND bearing health: [status]. Dominant fault is [fault_type]. Severity is [level]. Risk of motor winding contamination: [risk assessment].',
                    'data_type': 'motor_assessment'
                },
                {
                    'id': 'MOTOR_Q2',
                    'equipment': 'Motor',
                    'parameter': 'MDE',
                    'text': 'For the MOTOR - Drive End (MDE) bearing: Based on the assessment parameters, what is the specific bearing component damage pattern (outer race vs inner race vs rolling element)? How does MDE condition compare to MND?',
                    'expected_format': 'Motor MDE bearing shows [damage pattern]. Affected component: [component]. Comparison to MND: [MDE is worse/comparable/better]. The progression suggests [failure mode].',
                    'data_type': 'motor_assessment'
                },
                {
                    'id': 'MOTOR_Q3',
                    'equipment': 'Motor',
                    'parameter': 'Combined_Motor',
                    'text': 'For the complete MOTOR: What is the combined risk assessment for both bearings (MND + MDE)? Is there a synchronized failure risk or sequential failure risk?',
                    'expected_format': 'Motor combined assessment: [risk_level]. Both bearings show [synchronized/sequential] failure risk. The critical bearing is [MND/MDE]. Urgent action needed for [specific bearing].',
                    'data_type': 'motor_combined'
                },
                {
                    'id': 'MOTOR_Q4',
                    'equipment': 'Motor',
                    'parameter': 'Motor_Recommendation',
                    'text': 'For the MOTOR: What is the specific maintenance action recommendation? Include bearing replacement timeline, lubrication intervention, and inspection requirements before the bearing fails catastrophically.',
                    'expected_format': 'Motor maintenance recommendation: [action]. Bearing replacement required within [timeline]. Lubrication intervention: [yes/no/what type]. Pre-replacement inspection: [required checks]. Cost of inaction: [failure risk].',
                    'data_type': 'motor_recommendation'
                }
            ],
            'pump': [
                {
                    'id': 'PUMP_Q1',
                    'equipment': 'Pump',
                    'parameter': 'PND',
                    'text': 'For the PUMP - Non-Drive End (PND) bearing: Summarize the bearing health status including dominant fault type, severity, and potential for seal contamination or fluid leakage.',
                    'expected_format': 'Pump PND bearing health: [status]. Dominant fault is [fault_type]. Severity is [level]. Risk of seal/fluid contamination: [risk assessment].',
                    'data_type': 'pump_assessment'
                },
                {
                    'id': 'PUMP_Q2',
                    'equipment': 'Pump',
                    'parameter': 'Pump_Hydraulics',
                    'text': 'For the PUMP: How does bearing damage affect pump performance (flow rate, pressure stability, vibration transmission to fluid)? Will bearing failure impact downstream equipment?',
                    'expected_format': 'Pump bearing damage impacts: [performance effects]. Flow/pressure stability: [affected/stable]. Vibration transmission: [level]. Risk to downstream equipment: [yes/no - specify].',
                    'data_type': 'pump_impact'
                },
                {
                    'id': 'PUMP_Q3',
                    'equipment': 'Pump',
                    'parameter': 'Pump_Lubrication',
                    'text': 'For the PUMP: Is the bearing degradation due to lubrication issues, bearing quality, misalignment, or contamination? What preventive actions are needed?',
                    'expected_format': 'Root cause analysis: [primary cause]. Contributing factors: [list]. Preventive actions: [specific actions]. Lubrication quality check: [required]. Alignment verification: [required/not required].',
                    'data_type': 'pump_root_cause'
                },
                {
                    'id': 'PUMP_Q4',
                    'equipment': 'Pump',
                    'parameter': 'Pump_Recommendation',
                    'text': 'For the PUMP: What is the specific maintenance action recommendation? Include bearing replacement timeline, seal integrity checks, and post-maintenance validation steps.',
                    'expected_format': 'Pump maintenance recommendation: [action]. Bearing replacement required within [timeline]. Seal integrity check: [required checks]. Post-replacement validation: [steps]. Production impact: [downtime estimate].',
                    'data_type': 'pump_recommendation'
                }
            ]
        }

    def analyze_bearing_docx(self, docx_path: str) -> dict:
        """
        Analyze bearing assessment DOCX with 32-question structure
        
        Returns dict with observations, analysis, motor assessment, pump assessment
        """
        logger.info(f"Starting enhanced 32-question analysis: {docx_path}")
        
        try:
            # Step 1: Load document
            logger.info("Step 1: Loading DOCX...")
            file_id, metadata = self.document_ingestion.ingest_document(
                filepath=docx_path,
                vectorize=True
            )
            logger.info(f"DOCX loaded. File ID: {file_id}")
            
            # Step 2: Get all prompt sets
            observation_prompts = self._get_observation_prompts()
            analysis_prompts = self._get_analysis_prompts()
            equipment_prompts = self._get_equipment_assessment_prompts()
            
            # Step 3: Initialize results structure
            results = {
                'observations': {},
                'analysis': {},
                'motor_assessment': {},
                'pump_assessment': {},
                'metadata': {
                    'docx_path': docx_path,
                    'file_id': file_id,
                    'total_questions': 24, # 8+8+4+4
                    'status': 'in_progress'
                }
            }
            
            # Step 4: Run OBSERVATION PROMPTS (Q1-Q8)
            logger.info("Step 4: Running OBSERVATION PROMPTS (Q1-Q8)...")
            for prompt in observation_prompts:
                prompt_id = prompt['id']
                try:
                    # --- [FIX] RATE LIMITING ---
                    # Wait 6.1 seconds to stay under 10 requests/minute
                    time.sleep(6.1) 
                    # --- [END FIX] ---

                    logger.info(f"Running {prompt_id}...")
                    
                    robust_query = (
                        f"You are analyzing a bearing condition assessment report.\n\n"
                        f"STRICT INSTRUCTION: Extract ONLY numeric data and direct comparisons from the Parameter Assessment table.\n"
                        f"Do NOT speculate or interpret. Only report what is explicitly stated in tables.\n\n"
                        f"Expected answer format:\n'{prompt['expected_format']}'\n\n"
                        f"QUESTION: {prompt['text']}"
                    )
                    
                    # --- [FIX] ---
                    # Call _handle_document_query directly to bypass the intent parser.
                    doc_query_params = {
                        "query": robust_query,  # Use the full query for RAG
                        "file_id": file_id,
                        "top_k": 3  # Get 3 relevant chunks
                    }
                    result = self.analytical_agent._handle_document_query(
                        user_query=prompt['text'], # Use simple text for metadata
                        parameters=doc_query_params
                    )
                    # --- [END FIX] ---
                    
                    results['observations'][prompt_id] = {
                        'id': prompt_id,
                        'parameter': prompt['parameter'],
                        'question': prompt['text'],
                        'answer': result.get('narrative', 'No answer generated'),
                        'status': 'success',
                        'data_type': prompt['data_type']
                    }
                    logger.info(f"✅ {prompt_id} completed")
                    
                except Exception as e:
                    logger.error(f"❌ {prompt_id} failed: {e}")
                    results['observations'][prompt_id] = {
                        'id': prompt_id,
                        'parameter': prompt['parameter'],
                        'question': prompt['text'],
                        'answer': '',
                        'error': str(e),
                        'status': 'failed'
                    }
            
            # Step 5: Run ANALYSIS PROMPTS (Q1-Q8)
            logger.info("\nStep 5: Running ANALYSIS PROMPTS (Q1-Q8)...")
            for prompt in analysis_prompts:
                prompt_id = prompt['id']
                try:
                    # --- [FIX] RATE LIMITING ---
                    time.sleep(6.1)
                    # --- [END FIX] ---

                    logger.info(f"Running {prompt_id}...")
                    
                    # Analysis should reference observation if available
                    obs_reference = ""
                    obs_id = f"Q{prompt_id.split('_')[0][1:]}_OBS"
                    if obs_id in results['observations']:
                        obs_answer = results['observations'][obs_id].get('answer', '')
                        obs_reference = f"\n\nContext from observation: {obs_answer}\n"
                    
                    robust_query = (
                        f"You are analyzing a bearing condition assessment report.\n\n"
                        f"INSTRUCTION: Provide technical interpretation and analysis based on the numeric observations.\n"
                        f"Include fault type, failure mechanism, bearing component affected, and risk assessment.\n"
                        f"{obs_reference}"
                        f"Expected answer format:\n'{prompt['expected_format']}'\n\n"
                        f"QUESTION: {prompt['text']}"
                    )
                    
                    # --- [FIX] ---
                    doc_query_params = {
                        "query": robust_query,
                        "file_id": file_id,
                        "top_k": 3
                    }
                    result = self.analytical_agent._handle_document_query(
                        user_query=prompt['text'],
                        parameters=doc_query_params
                    )
                    # --- [END FIX] ---
                    
                    results['analysis'][prompt_id] = {
                        'id': prompt_id,
                        'parameter': prompt['parameter'],
                        'question': prompt['text'],
                        'answer': result.get('narrative', 'No answer generated'),
                        'status': 'success',
                        'data_type': prompt['data_type'],
                        'references_observation': obs_id
                    }
                    logger.info(f"✅ {prompt_id} completed")
                    
                except Exception as e:
                    logger.error(f"❌ {prompt_id} failed: {e}")
                    results['analysis'][prompt_id] = {
                        'id': prompt_id,
                        'parameter': prompt['parameter'],
                        'question': prompt['text'],
                        'answer': '',
                        'error': str(e),
                        'status': 'failed'
                    }
            
            # Step 6: Run MOTOR ASSESSMENT (4 questions)
            logger.info("\nStep 6: Running MOTOR ASSESSMENT (4 questions)...")
            for prompt in equipment_prompts['motor']:
                prompt_id = prompt['id']
                try:
                    # --- [FIX] RATE LIMITING ---
                    time.sleep(6.1)
                    # --- [END FIX] ---
                    
                    logger.info(f"Running {prompt_id}...")
                    
                    comprehensive_context = (
                        f"\n\nContext - All observations and analysis findings above have been extracted. "
                        f"Use them to inform the motor-specific assessment."
                    )
                    
                    robust_query = (
                        f"You are analyzing motor bearing condition assessment.\n\n"
                        f"INSTRUCTION: Provide motor-specific bearing health assessment for the {prompt['parameter']} location.\n"
                        f"Reference findings from all observations and analysis prompts.\n"
                        f"{comprehensive_context}"
                        f"Expected answer format:\n'{prompt['expected_format']}'\n\n"
                        f"QUESTION: {prompt['text']}"
                    )
                    
                    # --- [FIX] ---
                    doc_query_params = {
                        "query": robust_query,
                        "file_id": file_id,
                        "top_k": 3
                    }
                    result = self.analytical_agent._handle_document_query(
                        user_query=prompt['text'],
                        parameters=doc_query_params
                    )
                    # --- [END FIX] ---
                    
                    results['motor_assessment'][prompt_id] = {
                        'id': prompt_id,
                        'equipment': prompt['equipment'],
                        'parameter': prompt['parameter'],
                        'question': prompt['text'],
                        'answer': result.get('narrative', 'No answer generated'),
                        'status': 'success',
                        'data_type': prompt['data_type']
                    }
                    logger.info(f"✅ {prompt_id} completed")
                    
                except Exception as e:
                    logger.error(f"❌ {prompt_id} failed: {e}")
                    results['motor_assessment'][prompt_id] = {
                        'id': prompt_id,
                        'equipment': prompt['equipment'],
                        'parameter': prompt['parameter'],
                        'question': prompt['text'],
                        'answer': '',
                        'error': str(e),
                        'status': 'failed'
                    }
            
            # Step 7: Run PUMP ASSESSMENT (4 questions)
            logger.info("\nStep 7: Running PUMP ASSESSMENT (4 questions)...")
            for prompt in equipment_prompts['pump']:
                prompt_id = prompt['id']
                try:
                    # --- [FIX] RATE LIMITING ---
                    time.sleep(6.1)
                    # --- [END FIX] ---

                    logger.info(f"Running {prompt_id}...")
                    
                    comprehensive_context = (
                        f"\n\nContext - All observations, analysis, and motor findings have been extracted. "
                        f"Use them to inform the pump-specific assessment."
                    )
                    
                    robust_query = (
                        f"You are analyzing pump bearing condition assessment.\n\n"
                        f"INSTRUCTION: Provide pump-specific bearing health assessment for the {prompt['parameter']} location.\n"
                        f"Consider pump performance impact, seal integrity, and fluid contamination risks.\n"
                        f"{comprehensive_context}"
                        f"Expected answer format:\n'{prompt['expected_format']}'\n\n"
                        f"QUESTION: {prompt['text']}"
                    )
                    
                    # --- [FIX] ---
                    doc_query_params = {
                        "query": robust_query,
                        "file_id": file_id,
                        "top_k": 3
                    }
                    result = self.analytical_agent._handle_document_query(
                        user_query=prompt['text'],
                        parameters=doc_query_params
                    )
                    # --- [END FIX] ---
                    
                    results['pump_assessment'][prompt_id] = {
                        'id': prompt_id,
                        'equipment': prompt['equipment'],
                        'parameter': prompt['parameter'],
                        'question': prompt['text'],
                        'answer': result.get('narrative', 'No answer generated'),
                        'status': 'success',
                        'data_type': prompt['data_type']
                    }
                    logger.info(f"✅ {prompt_id} completed")
                    
                except Exception as e:
                    logger.error(f"❌ {prompt_id} failed: {e}")
                    results['pump_assessment'][prompt_id] = {
                        'id': prompt_id,
                        'equipment': prompt['equipment'],
                        'parameter': prompt['parameter'],
                        'question': prompt['text'],
                        'answer': '',
                        'error': str(e),
                        'status': 'failed'
                    }
            
            # Step 8: Cleanup
            logger.info("\nStep 8: Cleaning up...")
            self._cleanup(file_id)
            
            # Step 9: Update metadata
            results['metadata']['status'] = 'complete'
            results['metadata']['observations_count'] = len([r for r in results['observations'].values() if r.get('status') == 'success'])
            results['metadata']['analysis_count'] = len([r for r in results['analysis'].values() if r.get('status') == 'success'])
            results['metadata']['motor_count'] = len([r for r in results['motor_assessment'].values() if r.get('status') == 'success'])
            results['metadata']['pump_count'] = len([r for r in results['pump_assessment'].values() if r.get('status') == 'success'])
            
            logger.info("=" * 60)
            logger.info("✅ 24-QUESTION ANALYSIS COMPLETE!")
            logger.info(f"Observations: {results['metadata']['observations_count']}/8")
            logger.info(f"Analysis: {results['metadata']['analysis_count']}/8")
            logger.info(f"Motor Assessment: {results['metadata']['motor_count']}/4")
            logger.info(f"Pump Assessment: {results['metadata']['pump_count']}/4")
            logger.info("=" * 60)
            
            return {
                'success': True,
                'file_id': file_id,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def _cleanup(self, file_id: str):
        """Clean up vector DB files"""
        try:
            index_path = self.settings.get_vector_db_path(file_id)
            metadata_path = self.settings.get_metadata_path(file_id)
            
            if index_path.exists():
                index_path.unlink()
                logger.info(f"Deleted: {index_path}")
            
            if metadata_path.exists():
                metadata_path.unlink()
                logger.info(f"Deleted: {metadata_path}")
                
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


# Global instance
dashboard_api = EnhancedDashboardAnalysisAPI()