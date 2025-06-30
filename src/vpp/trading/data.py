"""
Market data providers for VPP trading operations.

This module provides market data feeds including simulated,
historical, and live data sources for energy markets.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging
import time
import threading
import queue


@dataclass
class PriceData:
    """Price data point."""
    timestamp: datetime
    market: str
    price: float
    volume: float = 0.0
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open_price: Optional[float] = None
    close_price: Optional[float] = None


@dataclass
class VolumeData:
    """Volume data point."""
    timestamp: datetime
    market: str
    volume: float
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    trade_count: int = 0


@dataclass
class MarketData:
    """Complete market data snapshot."""
    market: str
    timestamp: datetime
    last_price: Optional[float] = None
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    volume: float = 0.0
    high: Optional[float] = None
    low: Optional[float] = None
    open_price: Optional[float] = None
    close_price: Optional[float] = None
    
    # Market depth
    bid_levels: List[Tuple[float, float]] = field(default_factory=list)
    ask_levels: List[Tuple[float, float]] = field(default_factory=list)
    
    # System information
    total_demand: float = 0.0
    total_supply: float = 0.0
    clearing_price: Optional[float] = None
    system_lambda: Optional[float] = None
    frequency: float = 60.0
    
    # Metadata
    data_quality: str = "good"  # good, fair, poor
    source: str = "unknown"
    latency_ms: float = 0.0


class MarketDataProvider(ABC):
    """Base class for market data providers."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """Initialize market data provider."""
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"trading.data.{name}")
        
        # Provider state
        self.is_connected = False
        self.is_streaming = False
        self.last_update: Optional[datetime] = None
        
        # Data storage
        self.latest_data: Dict[str, MarketData] = {}
        self.price_history: Dict[str, List[PriceData]] = {}
        self.volume_history: Dict[str, List[VolumeData]] = {}
        
        # Configuration
        self.update_frequency = self.config.get("update_frequency", 60)  # seconds
        self.max_history_size = self.config.get("max_history_size", 1000)
        self.markets = self.config.get("markets", ["day_ahead", "real_time"])
        
        # Threading
        self._stop_event = threading.Event()
        self._data_thread: Optional[threading.Thread] = None
        self._data_queue = queue.Queue()
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to data source."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from data source."""
        pass
    
    @abstractmethod
    def _fetch_data(self, market: str) -> Optional[MarketData]:
        """Fetch data for a specific market."""
        pass
    
    def start_streaming(self) -> bool:
        """Start streaming market data."""
        if self.is_streaming:
            return True
        
        if not self.is_connected and not self.connect():
            return False
        
        self._stop_event.clear()
        self._data_thread = threading.Thread(target=self._data_loop, daemon=True)
        self._data_thread.start()
        
        self.is_streaming = True
        self.logger.info(f"Started streaming data for {self.name}")
        return True
    
    def stop_streaming(self) -> None:
        """Stop streaming market data."""
        if not self.is_streaming:
            return
        
        self._stop_event.set()
        if self._data_thread:
            self._data_thread.join(timeout=5.0)
        
        self.is_streaming = False
        self.logger.info(f"Stopped streaming data for {self.name}")
    
    def get_latest(self, market: str = None) -> Optional[MarketData]:
        """Get latest market data."""
        if market:
            return self.latest_data.get(market)
        
        # Return most recent data across all markets
        if not self.latest_data:
            return None
        
        latest_time = max(data.timestamp for data in self.latest_data.values())
        for data in self.latest_data.values():
            if data.timestamp == latest_time:
                return data
        
        return None
    
    def get_price_history(self, market: str, hours: int = 24) -> List[PriceData]:
        """Get price history for a market."""
        if market not in self.price_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [data for data in self.price_history[market] 
                if data.timestamp >= cutoff_time]
    
    def get_volume_history(self, market: str, hours: int = 24) -> List[VolumeData]:
        """Get volume history for a market."""
        if market not in self.volume_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [data for data in self.volume_history[market] 
                if data.timestamp >= cutoff_time]
    
    def _data_loop(self) -> None:
        """Main data fetching loop."""
        while not self._stop_event.is_set():
            try:
                for market in self.markets:
                    data = self._fetch_data(market)
                    if data:
                        self._update_data(data)
                
                time.sleep(self.update_frequency)
                
            except Exception as e:
                self.logger.error(f"Error in data loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _update_data(self, data: MarketData) -> None:
        """Update internal data storage."""
        market = data.market
        
        # Update latest data
        self.latest_data[market] = data
        self.last_update = data.timestamp
        
        # Update price history
        if market not in self.price_history:
            self.price_history[market] = []
        
        price_data = PriceData(
            timestamp=data.timestamp,
            market=market,
            price=data.last_price or 0.0,
            volume=data.volume,
            bid=data.bid_price,
            ask=data.ask_price,
            high=data.high,
            low=data.low,
            open_price=data.open_price,
            close_price=data.close_price
        )
        
        self.price_history[market].append(price_data)
        
        # Update volume history
        if market not in self.volume_history:
            self.volume_history[market] = []
        
        volume_data = VolumeData(
            timestamp=data.timestamp,
            market=market,
            volume=data.volume
        )
        
        self.volume_history[market].append(volume_data)
        
        # Limit history size
        if len(self.price_history[market]) > self.max_history_size:
            self.price_history[market] = self.price_history[market][-self.max_history_size:]
        
        if len(self.volume_history[market]) > self.max_history_size:
            self.volume_history[market] = self.volume_history[market][-self.max_history_size:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get provider statistics."""
        return {
            "name": self.name,
            "is_connected": self.is_connected,
            "is_streaming": self.is_streaming,
            "last_update": self.last_update,
            "markets_count": len(self.markets),
            "data_points": sum(len(history) for history in self.price_history.values()),
            "update_frequency": self.update_frequency
        }


class SimulatedDataProvider(MarketDataProvider):
    """Simulated market data provider for testing and development."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize simulated data provider."""
        super().__init__("simulated", config)
        
        # Simulation parameters
        self.base_prices = self.config.get("base_prices", {
            "day_ahead": 0.10,
            "real_time": 0.12,
            "ancillary_services": 50.0,
            "bilateral": 0.11
        })
        
        self.volatility = self.config.get("volatility", 0.1)  # 10% volatility
        self.trend = self.config.get("trend", 0.0)  # No trend by default
        self.seasonal_amplitude = self.config.get("seasonal_amplitude", 0.2)
        
        # Random number generator
        self.rng = np.random.RandomState(self.config.get("seed", 42))
        
        # Simulation state
        self.simulation_time = datetime.now()
        self.price_paths: Dict[str, List[float]] = {}
    
    def connect(self) -> bool:
        """Connect to simulated data source."""
        self.is_connected = True
        self.logger.info("Connected to simulated data source")
        
        # Initialize price paths
        for market in self.markets:
            self.price_paths[market] = [self.base_prices.get(market, 0.10)]
        
        return True
    
    def disconnect(self) -> None:
        """Disconnect from simulated data source."""
        self.is_connected = False
        self.logger.info("Disconnected from simulated data source")
    
    def _fetch_data(self, market: str) -> Optional[MarketData]:
        """Generate simulated market data."""
        if not self.is_connected:
            return None
        
        try:
            current_time = datetime.now()
            
            # Generate price using geometric Brownian motion with seasonality
            if market not in self.price_paths:
                self.price_paths[market] = [self.base_prices.get(market, 0.10)]
            
            last_price = self.price_paths[market][-1]
            
            # Seasonal component (daily pattern)
            hour = current_time.hour
            seasonal_factor = 1 + self.seasonal_amplitude * np.sin(2 * np.pi * (hour - 6) / 24)
            
            # Random walk component
            dt = 1.0 / (24 * 60)  # 1 minute time step
            drift = self.trend * dt
            diffusion = self.volatility * np.sqrt(dt) * self.rng.normal()
            
            # New price
            new_price = last_price * seasonal_factor * np.exp(drift + diffusion)
            new_price = max(0.01, new_price)  # Ensure positive price
            
            self.price_paths[market].append(new_price)
            
            # Limit price path length
            if len(self.price_paths[market]) > 1000:
                self.price_paths[market] = self.price_paths[market][-1000:]
            
            # Generate bid/ask spread
            spread_pct = self.rng.uniform(0.001, 0.01)  # 0.1% to 1% spread
            spread = new_price * spread_pct
            
            bid_price = new_price - spread / 2
            ask_price = new_price + spread / 2
            
            # Generate volume
            base_volume = self.config.get("base_volume", 100.0)
            volume = base_volume * self.rng.lognormal(0, 0.5)
            
            # Generate market depth
            bid_levels = []
            ask_levels = []
            
            for i in range(5):  # 5 levels of depth
                level_volume = volume * self.rng.uniform(0.1, 0.5)
                bid_levels.append((bid_price - i * spread * 0.1, level_volume))
                ask_levels.append((ask_price + i * spread * 0.1, level_volume))
            
            # System data for real-time market
            if market == "real_time":
                system_demand = 1000 + 200 * np.sin(2 * np.pi * hour / 24) + self.rng.normal(0, 50)
                system_supply = system_demand + self.rng.normal(0, 20)
                frequency = 60.0 + self.rng.normal(0, 0.1)
            else:
                system_demand = 0.0
                system_supply = 0.0
                frequency = 60.0
            
            # Create market data
            data = MarketData(
                market=market,
                timestamp=current_time,
                last_price=new_price,
                bid_price=bid_price,
                ask_price=ask_price,
                volume=volume,
                high=new_price * 1.02,
                low=new_price * 0.98,
                open_price=last_price,
                close_price=new_price,
                bid_levels=bid_levels,
                ask_levels=ask_levels,
                total_demand=system_demand,
                total_supply=system_supply,
                clearing_price=new_price,
                frequency=frequency,
                data_quality="good",
                source="simulated",
                latency_ms=self.rng.uniform(1, 10)
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error generating simulated data for {market}: {e}")
            return None


class LiveDataProvider(MarketDataProvider):
    """Live market data provider for real market feeds."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize live data provider."""
        super().__init__("live", config)
        
        # Connection parameters
        self.api_endpoint = self.config.get("api_endpoint", "")
        self.api_key = self.config.get("api_key", "")
        self.timeout = self.config.get("timeout", 30)
        
        # Rate limiting
        self.rate_limit = self.config.get("rate_limit", 60)  # requests per minute
        self.last_request_time = 0.0
        
        # Connection state
        self.session = None
    
    def connect(self) -> bool:
        """Connect to live data source."""
        try:
            # This is a placeholder - implement actual API connection
            # import requests
            # self.session = requests.Session()
            # self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
            
            self.is_connected = True
            self.logger.info(f"Connected to live data source: {self.api_endpoint}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to live data source: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from live data source."""
        if self.session:
            self.session.close()
            self.session = None
        
        self.is_connected = False
        self.logger.info("Disconnected from live data source")
    
    def _fetch_data(self, market: str) -> Optional[MarketData]:
        """Fetch live market data."""
        if not self.is_connected:
            return None
        
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            min_interval = 60.0 / self.rate_limit
            
            if time_since_last < min_interval:
                time.sleep(min_interval - time_since_last)
            
            self.last_request_time = time.time()
            
            # This is a placeholder - implement actual API call
            # response = self.session.get(
            #     f"{self.api_endpoint}/market/{market}",
            #     timeout=self.timeout
            # )
            # response.raise_for_status()
            # api_data = response.json()
            
            # For now, return simulated data as placeholder
            return self._generate_placeholder_data(market)
            
        except Exception as e:
            self.logger.error(f"Error fetching live data for {market}: {e}")
            return None
    
    def _generate_placeholder_data(self, market: str) -> MarketData:
        """Generate placeholder data until real API is implemented."""
        # This is temporary - replace with actual API data parsing
        current_time = datetime.now()
        
        # Simple price simulation
        base_price = {"day_ahead": 0.10, "real_time": 0.12}.get(market, 0.10)
        price = base_price * (1 + 0.1 * np.sin(current_time.hour * np.pi / 12))
        
        return MarketData(
            market=market,
            timestamp=current_time,
            last_price=price,
            bid_price=price * 0.999,
            ask_price=price * 1.001,
            volume=100.0,
            data_quality="good",
            source="live_api",
            latency_ms=50.0
        )


class HistoricalDataProvider(MarketDataProvider):
    """Historical market data provider for backtesting."""
    
    def __init__(self, data_file: str, config: Dict[str, Any] = None):
        """Initialize historical data provider."""
        super().__init__("historical", config)
        self.data_file = data_file
        self.historical_data: Dict[str, List[MarketData]] = {}
        self.current_index = 0
        self.playback_speed = self.config.get("playback_speed", 1.0)  # 1.0 = real-time
    
    def connect(self) -> bool:
        """Load historical data."""
        try:
            # This is a placeholder - implement actual data loading
            # import pandas as pd
            # df = pd.read_csv(self.data_file)
            # self.historical_data = self._parse_historical_data(df)
            
            self.is_connected = True
            self.logger.info(f"Loaded historical data from {self.data_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load historical data: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from historical data."""
        self.is_connected = False
        self.historical_data.clear()
        self.current_index = 0
    
    def _fetch_data(self, market: str) -> Optional[MarketData]:
        """Fetch next historical data point."""
        if not self.is_connected or market not in self.historical_data:
            return None
        
        data_series = self.historical_data[market]
        if self.current_index >= len(data_series):
            return None  # End of data
        
        data = data_series[self.current_index]
        self.current_index += 1
        
        return data
    
    def reset_playback(self) -> None:
        """Reset playback to beginning."""
        self.current_index = 0
    
    def seek_to_time(self, target_time: datetime) -> bool:
        """Seek to specific time in historical data."""
        # Implementation would search for closest timestamp
        return True


def create_data_provider(provider_type: str, **kwargs) -> MarketDataProvider:
    """
    Factory function to create data providers.
    
    Args:
        provider_type: Type of provider ("simulated", "live", "historical")
        **kwargs: Provider configuration
        
    Returns:
        MarketDataProvider instance
        
    Example:
        >>> provider = create_data_provider("simulated", volatility=0.2)
        >>> provider = create_data_provider("live", api_endpoint="https://api.example.com")
    """
    provider_type = provider_type.lower()
    
    if provider_type == "simulated":
        return SimulatedDataProvider(kwargs)
    elif provider_type == "live":
        return LiveDataProvider(kwargs)
    elif provider_type == "historical":
        data_file = kwargs.pop("data_file", "historical_data.csv")
        return HistoricalDataProvider(data_file, kwargs)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


class DataAggregator:
    """Aggregates data from multiple providers."""
    
    def __init__(self):
        """Initialize data aggregator."""
        self.providers: Dict[str, MarketDataProvider] = {}
        self.logger = logging.getLogger("trading.data.aggregator")
    
    def add_provider(self, provider: MarketDataProvider) -> None:
        """Add a data provider."""
        self.providers[provider.name] = provider
        self.logger.info(f"Added data provider: {provider.name}")
    
    def start_all(self) -> None:
        """Start all providers."""
        for provider in self.providers.values():
            provider.start_streaming()
    
    def stop_all(self) -> None:
        """Stop all providers."""
        for provider in self.providers.values():
            provider.stop_streaming()
    
    def get_latest_data(self) -> Dict[str, MarketData]:
        """Get latest data from all providers."""
        latest_data = {}
        
        for provider in self.providers.values():
            for market in provider.markets:
                data = provider.get_latest(market)
                if data:
                    # Use most recent data if multiple providers have same market
                    if (market not in latest_data or 
                        data.timestamp > latest_data[market].timestamp):
                        latest_data[market] = data
        
        return latest_data
    
    def get_provider_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all providers."""
        return {name: provider.get_statistics() 
                for name, provider in self.providers.items()}
