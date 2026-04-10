"""
Audio Synchronization Module for Music Video Generation

Extracts beat, tempo, and rhythm features from audio to synchronize
video motion with music.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, List
from pathlib import Path


class AudioFeatureExtractor:
    """
    Extract audio features for video synchronization.
    
    Uses librosa for beat detection and feature extraction.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        n_mfcc: int = 13,
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            return audio, sr
        except ImportError:
            raise ImportError("Please install librosa: pip install librosa")
    
    def extract_beats(self, audio: np.ndarray) -> np.ndarray:
        """Detect beat positions."""
        import librosa
        tempo, beats = librosa.beat.beat_track(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )
        return beats
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract comprehensive audio features."""
        import librosa
        
        # MFCCs (timbre)
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
        )
        
        # Spectral contrast (texture)
        contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )
        
        # Chroma (harmony)
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )
        
        # RMS energy (loudness)
        rms = librosa.feature.rms(
            y=audio,
            hop_length=self.hop_length,
        )
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            hop_length=self.hop_length,
        )
        
        # Beat tracking
        tempo, beats = librosa.beat.beat_track(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )
        
        return {
            'mfccs': mfccs,
            'contrast': contrast,
            'chroma': chroma,
            'rms': rms,
            'zcr': zcr,
            'tempo': np.array([tempo]),
            'beats': beats,
        }
    
    def create_motion_signal(
        self,
        features: Dict[str, np.ndarray],
        duration_sec: float,
        fps: int = 24,
    ) -> np.ndarray:
        """
        Create per-frame motion intensity signal from audio features.
        
        Combines RMS energy, beat information, and spectral features
        into a single motion control signal.
        """
        num_frames = int(duration_sec * fps)
        
        # Use RMS energy as base motion signal
        rms = features['rms'][0]
        
        # Upsample to match frame count
        from scipy.interpolate import interp1d
        
        time_audio = np.linspace(0, duration_sec, len(rms))
        time_video = np.linspace(0, duration_sec, num_frames)
        
        interp_func = interp1d(time_audio, rms, kind='linear', fill_value='extrapolate')
        motion_signal = interp_func(time_video)
        
        # Add beat emphasis
        beats = features['beats']
        beat_frames = (beats / self.sample_rate * self.hop_length).astype(int)
        beat_frames = beat_frames[beat_frames < num_frames]
        
        # Boost motion at beat positions
        for bf in beat_frames:
            start = max(0, bf - 2)
            end = min(num_frames, bf + 2)
            motion_signal[start:end] *= 1.5
        
        # Normalize to [0, 1]
        motion_signal = (motion_signal - motion_signal.min()) / (motion_signal.max() - motion_signal.min() + 1e-8)
        
        return motion_signal


class AudioEncoder(nn.Module):
    """
    Neural audio encoder for deep audio features.
    
    Can be used for more sophisticated audio-video synchronization.
    """
    
    def __init__(
        self,
        input_dim: int = 13,  # MFCC dimension
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode audio features.
        
        Args:
            x: [B, T, input_dim] - audio features over time
        
        Returns:
            encoded: [B, T, output_dim] - encoded representations
        """
        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.output_proj(x)
        x = self.norm(x)
        return x


class AudioVideoSyncModule(nn.Module):
    """
    Cross-modal attention module for audio-video synchronization.
    
    Aligns video motion embeddings with audio features.
    """
    
    def __init__(
        self,
        audio_dim: int = 768,
        video_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Projections to common space
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        
        # Cross-attention from video to audio
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, video_dim)
        self.norm = nn.LayerNorm(video_dim)
    
    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
        sync_strength: float = 0.5,
    ) -> torch.Tensor:
        """
        Synchronize video features with audio.
        
        Args:
            video_features: [B, T, video_dim] - video latent features
            audio_features: [B, T, audio_dim] - audio embeddings
            sync_strength: How strongly to couple audio and video (0.0-1.0)
        
        Returns:
            synced_features: [B, T, video_dim] - audio-synchronized video features
        """
        # Project to common space
        audio_proj = self.audio_proj(audio_features)
        video_proj = self.video_proj(video_features)
        
        # Cross-attention: video queries, audio keys/values
        attn_out, _ = self.cross_attn(
            query=video_proj,
            key=audio_proj,
            value=audio_proj,
        )
        
        # Blend with original features based on sync strength
        blended = video_proj + sync_strength * attn_out
        
        # Project back to video space
        out = self.out_proj(blended)
        out = self.norm(out)
        
        return out


def prepare_audio_for_generation(
    audio_path: str,
    duration_sec: float,
    fps: int = 24,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Prepare audio features for video generation.
    
    Args:
        audio_path: Path to audio file
        duration_sec: Duration of video to generate
        fps: Frames per second
        device: Target device
    
    Returns:
        motion_signal: Per-frame motion intensity [T]
        audio_embeddings: Deep audio features [1, T, dim]
        metadata: Additional audio info (tempo, etc.)
    """
    extractor = AudioFeatureExtractor()
    
    # Load and extract features
    audio, sr = extractor.load_audio(audio_path)
    features = extractor.extract_features(audio)
    
    # Create motion signal
    motion_signal = extractor.create_motion_signal(features, duration_sec, fps)
    
    # Prepare deep embeddings
    mfccs = torch.FloatTensor(features['mfccs'].T).unsqueeze(0).to(device)  # [1, T, 13]
    
    # Encode with neural network
    audio_encoder = AudioEncoder(input_dim=13, output_dim=768).to(device)
    audio_embeddings = audio_encoder(mfccs)
    
    metadata = {
        'tempo': float(features['tempo'][0]),
        'duration': duration_sec,
        'num_frames': int(duration_sec * fps),
        'sample_rate': sr,
    }
    
    return (
        torch.FloatTensor(motion_signal).to(device),
        audio_embeddings,
        metadata,
    )


if __name__ == "__main__":
    # Test audio processing
    import sys
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        motion, embeddings, meta = prepare_audio_for_generation(
            audio_path, duration_sec=8.0, fps=24
        )
        
        print(f"Tempo: {meta['tempo']} BPM")
        print(f"Duration: {meta['duration']}s")
        print(f"Frames: {meta['num_frames']}")
        print(f"Motion signal shape: {motion.shape}")
        print(f"Audio embeddings shape: {embeddings.shape}")
    else:
        print("Usage: python audio_sync.py <audio_file.mp3>")
