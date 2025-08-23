import numpy as np
import pandas as pd
import time
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    CNN = "cnn"
    RANDOM_FOREST = "random_forest"

@dataclass
class DeviceResources:
    """Track device resource states"""
    battery_level: float  # 0-100%
    memory_usage: float   # MB
    bandwidth_up: float   # Mbps
    bandwidth_down: float # Mbps
    is_active: bool = True

@dataclass
class TrainingMetrics:
    """Metrics collected during training"""
    round_id: int
    device_id: str
    battery_consumed: float
    memory_peak: float
    bandwidth_up_used: float
    bandwidth_down_used: float
    training_time: float
    communication_time: float

class IoTDevice:
    """Simulates an IoT device with ML capabilities"""
    
    def __init__(self, device_id: str, model_type: ModelType, data_size: int = 100):
        self.device_id = device_id
        self.model_type = model_type
        self.resources = DeviceResources(
            battery_level=random.uniform(20, 100),
            memory_usage=random.uniform(50, 200),
            bandwidth_up=random.uniform(1, 10),
            bandwidth_down=random.uniform(5, 50)
        )
        
        # Generate private dataset
        X, y = make_classification(
            n_samples=data_size,
            n_features=10,
            n_classes=2,
            random_state=hash(device_id) % 2**32
        )
        self.private_data = (X, y)
        
        # Initialize model
        if model_type == ModelType.RANDOM_FOREST:
            self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        else:
            # Simplified CNN representation (using RF for demo)
            self.model = RandomForestClassifier(n_estimators=20, random_state=42)
    
    def can_participate(self) -> bool:
        """Check if device can participate based on resources"""
        return (self.resources.battery_level > 15 and 
                self.resources.memory_usage < 300 and
                self.resources.is_active)
    
    def local_training(self, global_update: Optional[Dict] = None) -> Tuple[Dict, TrainingMetrics]:
        """Perform local training and return update with metrics"""
        start_time = time.time()
        initial_battery = self.resources.battery_level
        initial_memory = self.resources.memory_usage
        
        # Simulate receiving global update (bandwidth consumption)
        comm_start = time.time()
        if global_update:
            download_size = len(str(global_update)) / 1024  # KB
            download_time = download_size / (self.resources.bandwidth_down * 1024)  # seconds
            time.sleep(min(download_time / 1000, 0.1))  # Simulate download delay
        comm_time = time.time() - comm_start
        
        # Simulate local training
        X, y = self.private_data
        self.model.fit(X, y)
        
        # Simulate resource consumption during training
        battery_drain = random.uniform(2, 8)  # 2-8% battery per training
        memory_spike = random.uniform(20, 80)  # 20-80 MB memory spike
        
        self.resources.battery_level = max(0, self.resources.battery_level - battery_drain)
        peak_memory = self.resources.memory_usage + memory_spike
        
        training_time = time.time() - start_time
        
        # Create local update (model parameters simulation)
        if hasattr(self.model, 'feature_importances_'):
            local_update = {
                'device_id': self.device_id,
                'feature_importances': self.model.feature_importances_.tolist(),
                'n_estimators': getattr(self.model, 'n_estimators', 10),
                'accuracy': self.model.score(X, y)
            }
        else:
            local_update = {
                'device_id': self.device_id,
                'weights': np.random.rand(10).tolist(),  # Simulated weights
                'accuracy': random.uniform(0.7, 0.95)
            }
        
        # Simulate upload bandwidth consumption
        upload_size = len(str(local_update)) / 1024  # KB
        upload_bandwidth_used = upload_size / max(training_time, 0.1)  # KB/s
        
        metrics = TrainingMetrics(
            round_id=0,  # Will be set by gateway
            device_id=self.device_id,
            battery_consumed=battery_drain,
            memory_peak=peak_memory,
            bandwidth_up_used=upload_bandwidth_used,
            bandwidth_down_used=download_size / max(comm_time, 0.01),
            training_time=training_time,
            communication_time=comm_time
        )
        
        return local_update, metrics
    
    def get_selection_features(self) -> Dict:
        """Return features for gateway's device selection decision tree"""
        return {
            'battery_level': self.resources.battery_level,
            'memory_available': 400 - self.resources.memory_usage,  # Assume 400MB total
            'bandwidth_up': self.resources.bandwidth_up,
            'bandwidth_down': self.resources.bandwidth_down,
            'is_active': int(self.resources.is_active)
        }

class Gateway:
    """Gateway with decision tree for device selection"""
    
    def __init__(self, devices: List[IoTDevice]):
        self.devices = {device.device_id: device for device in devices}
        self.global_update = {}
        self.local_updates_history = []
        self.metrics_history = []
        
        # Initialize decision tree for device selection
        self.selection_tree = DecisionTreeClassifier(random_state=42)
        self._train_selection_model()
        
    def _train_selection_model(self):
        """Train the decision tree for device selection"""
        # Generate training data for device selection
        # Features: battery, memory_available, bandwidth_up, bandwidth_down, is_active
        # Target: should_select (1 if device should be selected, 0 otherwise)
        
        training_features = []
        training_targets = []
        
        # Generate synthetic training data for device selection
        for _ in range(1000):
            battery = random.uniform(0, 100)
            memory_avail = random.uniform(0, 300)
            bw_up = random.uniform(0.5, 15)
            bw_down = random.uniform(1, 60)
            is_active = random.choice([0, 1])
            
            # Selection logic: good battery + sufficient memory + decent bandwidth + active
            should_select = int(
                battery > 20 and 
                memory_avail > 50 and 
                bw_up > 2 and 
                is_active == 1
            )
            
            training_features.append([battery, memory_avail, bw_up, bw_down, is_active])
            training_targets.append(should_select)
        
        self.selection_tree.fit(training_features, training_targets)
        logger.info("Device selection model trained")
    
    def select_devices(self, max_devices: int = None) -> List[str]:
        """Use decision tree to select devices for training"""
        eligible_devices = []
        
        for device_id, device in self.devices.items():
            if device.can_participate():
                features = device.get_selection_features()
                feature_vector = [
                    features['battery_level'],
                    features['memory_available'],
                    features['bandwidth_up'],
                    features['bandwidth_down'],
                    features['is_active']
                ]
                
                # Use decision tree to predict if device should be selected
                should_select = self.selection_tree.predict([feature_vector])[0]
                selection_probability = self.selection_tree.predict_proba([feature_vector])[0][1]
                
                if should_select and selection_probability > 0.6:
                    eligible_devices.append((device_id, selection_probability))
        
        # Sort by selection probability and limit by max_devices
        eligible_devices.sort(key=lambda x: x[1], reverse=True)
        if max_devices:
            eligible_devices = eligible_devices[:max_devices]
        
        selected = [device_id for device_id, _ in eligible_devices]
        logger.info(f"Selected {len(selected)} devices: {selected}")
        return selected
    
    def federated_round(self, round_id: int, max_devices: int = None) -> List[TrainingMetrics]:
        """Execute one round of federated learning"""
        logger.info(f"Starting round {round_id}")
        
        # Select devices using decision tree
        selected_devices = self.select_devices(max_devices)
        
        if not selected_devices:
            logger.warning("No devices selected for training")
            return []
        
        round_metrics = []
        local_updates = []
        
        # Send global update to selected devices and collect local updates
        for device_id in selected_devices:
            device = self.devices[device_id]
            try:
                local_update, metrics = device.local_training(self.global_update)
                metrics.round_id = round_id
                
                local_updates.append(local_update)
                round_metrics.append(metrics)
                
                logger.info(f"Device {device_id} completed training - "
                          f"Battery: {device.resources.battery_level:.1f}%, "
                          f"Memory peak: {metrics.memory_peak:.1f}MB")
                
            except Exception as e:
                logger.error(f"Training failed for device {device_id}: {e}")
        
        # Store local updates (no aggregation as specified)
        self.local_updates_history.extend(local_updates)
        self.metrics_history.extend(round_metrics)
        
        # Update global update (in real scenario, this might come from external source)
        # For simulation, we'll create a dummy global update
        self.global_update = {
            'round': round_id,
            'timestamp': time.time(),
            'participating_devices': selected_devices
        }
        
        return round_metrics

class FederatedLearningSimulation:
    """Main simulation controller"""
    
    def __init__(self, num_devices: int = 10, model_type: ModelType = ModelType.RANDOM_FOREST):
        self.devices = [
            IoTDevice(f"device_{i}", model_type, data_size=random.randint(50, 200))
            for i in range(num_devices)
        ]
        self.gateway = Gateway(self.devices)
        self.simulation_results = pd.DataFrame()
    
    def run_simulation(self, num_rounds: int = 5, max_devices_per_round: int = 5) -> pd.DataFrame:
        """Run the complete federated learning simulation"""
        logger.info(f"Starting simulation: {num_rounds} rounds, max {max_devices_per_round} devices per round")
        
        all_metrics = []
        
        for round_id in range(num_rounds):
            round_metrics = self.gateway.federated_round(round_id, max_devices_per_round)
            all_metrics.extend(round_metrics)
            
            # Simulate some devices going offline or battery depletion
            if round_id > 0:
                for device in self.devices:
                    if device.resources.battery_level < 10:
                        device.resources.is_active = False
                        logger.info(f"Device {device.device_id} went offline (low battery)")
        
        # Convert metrics to DataFrame
        if all_metrics:
            self.simulation_results = pd.DataFrame([
                {
                    'round_id': m.round_id,
                    'device_id': m.device_id,
                    'battery_consumed': m.battery_consumed,
                    'memory_peak': m.memory_peak,
                    'bandwidth_up_used': m.bandwidth_up_used,
                    'bandwidth_down_used': m.bandwidth_down_used,
                    'training_time': m.training_time,
                    'communication_time': m.communication_time
                }
                for m in all_metrics
            ])
        
        return self.simulation_results
    
    def plot_results(self):
        """Visualize simulation results"""
        if self.simulation_results.empty:
            logger.warning("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Battery consumption by round
        battery_by_round = self.simulation_results.groupby('round_id')['battery_consumed'].agg(['mean', 'std'])
        axes[0, 0].errorbar(battery_by_round.index, battery_by_round['mean'], 
                           yerr=battery_by_round['std'], marker='o')
        axes[0, 0].set_title('Battery Consumption per Round')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Battery Consumed (%)')
        
        # Memory usage distribution
        axes[0, 1].hist(self.simulation_results['memory_peak'], bins=20, alpha=0.7)
        axes[0, 1].set_title('Memory Peak Usage Distribution')
        axes[0, 1].set_xlabel('Memory Peak (MB)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Bandwidth usage over time
        axes[1, 0].scatter(self.simulation_results['round_id'], 
                          self.simulation_results['bandwidth_up_used'], 
                          alpha=0.6, label='Upload')
        axes[1, 0].scatter(self.simulation_results['round_id'], 
                          self.simulation_results['bandwidth_down_used'], 
                          alpha=0.6, label='Download')
        axes[1, 0].set_title('Bandwidth Usage per Round')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Bandwidth (KB/s)')
        axes[1, 0].legend()
        
        # Training vs Communication time
        axes[1, 1].scatter(self.simulation_results['training_time'], 
                          self.simulation_results['communication_time'], 
                          alpha=0.6)
        axes[1, 1].set_title('Training vs Communication Time')
        axes[1, 1].set_xlabel('Training Time (s)')
        axes[1, 1].set_ylabel('Communication Time (s)')
        
        plt.tight_layout()
        plt.show()
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of the simulation"""
        if self.simulation_results.empty:
            return {}
        
        return {
            'total_rounds': self.simulation_results['round_id'].nunique(),
            'total_devices': self.simulation_results['device_id'].nunique(),
            'avg_battery_per_round': self.simulation_results['battery_consumed'].mean(),
            'avg_memory_peak': self.simulation_results['memory_peak'].mean(),
            'avg_training_time': self.simulation_results['training_time'].mean(),
            'avg_communication_time': self.simulation_results['communication_time'].mean(),
            'total_bandwidth_up': self.simulation_results['bandwidth_up_used'].sum(),
            'total_bandwidth_down': self.simulation_results['bandwidth_down_used'].sum()
        }

# Example usage
if __name__ == "__main__":
    # Create and run simulation
    sim = FederatedLearningSimulation(num_devices=15, model_type=ModelType.CNN)
    results = sim.run_simulation(num_rounds=100, max_devices_per_round=25)
    
    # Display results
    print("\n=== Simulation Results ===")
    print(results.head(10))
    
    print("\n=== Summary Statistics ===")
    stats = sim.get_summary_statistics()
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    # Plot results
    sim.plot_results()