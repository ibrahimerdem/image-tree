"""
Quick reference examples for using IntegratedConditionalGenerator
"""

import torch
import torch.nn as nn
import torch.optim as optim
from models.integrated_conditional_generator import (
    IntegratedConditionalGenerator,
    TrainingModule
)


# ============================================================
# EXAMPLE 1: Minimal Setup - Noise + Conditions
# ============================================================
def example_minimal():
    """Simplest use case: generate from noise and conditions"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Minimal Setup")
    print("="*60)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = IntegratedConditionalGenerator(
        num_conditions=9,
        noise_dim=100,
        output_size=512,
        device=device
    ).to(device)
    
    # Generate samples
    batch_size = 4
    model.eval()
    with torch.no_grad():
        conditions = torch.randn(batch_size, 9, device=device)
        images = model(conditions)  # Random noise is generated internally
    
    print(f"Output shape: {images.shape}")
    print(f"Output range: [{images.min():.2f}, {images.max():.2f}]")


# ============================================================
# EXAMPLE 2: Training Loop
# ============================================================
def example_training():
    """Complete training loop with loss computation"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Training Loop")
    print("="*60)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = IntegratedConditionalGenerator(
        num_conditions=9,
        use_vae=True,
        device=device
    ).to(device)
    
    # Training module with losses
    trainer = TrainingModule(
        model,
        reconstruction_weight=0.2,
        l1_weight=2.0,
        perceptual_weight=0.8,
        vae_kl_weight=0.1,
        device=device
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # Dummy batch
    batch_size = 2
    conditions = torch.randn(batch_size, 9, device=device)
    target = torch.randn(batch_size, 3, 512, 512, device=device)
    
    # Training step
    model.train()
    losses = trainer(conditions, target)
    loss_total = losses['total']
    
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()
    
    print("Training step completed!")
    print(f"Loss breakdown:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")


# ============================================================
# EXAMPLE 3: Image-to-Image with Pretrained Encoder
# ============================================================
def example_image_to_image():
    """Generate target from initial image + conditions"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Image-to-Image Generation")
    print("="*60)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Model with image encoder support
    model = IntegratedConditionalGenerator(
        num_conditions=9,
        initial_image=True,
        image_encoder_path='encoder_epoch_50.pth',  # Optional: path to encoder
        use_vae=True,
        device=device
    ).to(device)
    
    # Freeze encoder if available
    model.freeze_image_encoder()
    
    # Generation
    batch_size = 2
    model.eval()
    with torch.no_grad():
        initial = torch.randn(batch_size, 3, 128, 128, device=device)
        conditions = torch.randn(batch_size, 9, device=device)
        target = model(conditions, initial_image=initial)
    
    print(f"Initial image shape: {initial.shape}")
    print(f"Generated shape: {target.shape}")


# ============================================================
# EXAMPLE 4: Deterministic Generation (Fixed Noise)
# ============================================================
def example_fixed_noise():
    """Generate multiple samples with same conditions but different noise"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Fixed Noise for Consistency")
    print("="*60)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = IntegratedConditionalGenerator(
        num_conditions=9,
        noise_dim=100,
        device=device
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        # Fixed conditions
        conditions = torch.randn(1, 9, device=device)
        
        # Generate 3 samples with different noise
        outputs = []
        for i in range(3):
            noise = torch.randn(1, 100, device=device)
            output = model(conditions, noise=noise)
            outputs.append(output)
        
        # Or use same noise for deterministic output
        fixed_noise = torch.randn(1, 100, device=device)
        deterministic = model(conditions, noise=fixed_noise)
    
    print(f"Generated {len(outputs)} different samples")
    print(f"Deterministic output shape: {deterministic.shape}")


# ============================================================
# EXAMPLE 5: VAE with KL Divergence
# ============================================================
def example_vae():
    """Training with VAE latent space and KL divergence"""
    print("\n" + "="*60)
    print("EXAMPLE 5: VAE Training")
    print("="*60)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Model with VAE
    model = IntegratedConditionalGenerator(
        num_conditions=9,
        use_vae=True,
        latent_dim=256,
        device=device
    ).to(device)
    
    trainer = TrainingModule(model, vae_kl_weight=0.1, device=device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # Training step
    batch_size = 2
    conditions = torch.randn(batch_size, 9, device=device)
    target = torch.randn(batch_size, 3, 512, 512, device=device)
    
    model.train()
    losses = trainer(conditions, target)
    
    optimizer.zero_grad()
    losses['total'].backward()
    optimizer.step()
    
    print("VAE training step completed!")
    if 'kl' in losses:
        print(f"KL divergence: {losses['kl'].item():.4f}")
    print(f"Reconstruction loss: {losses['l1'].item():.4f}")


# ============================================================
# EXAMPLE 6: Gradient Accumulation for Larger Batch
# ============================================================
def example_gradient_accumulation():
    """Training with gradient accumulation for effective batch size"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Gradient Accumulation")
    print("="*60)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = IntegratedConditionalGenerator(
        num_conditions=9,
        device=device
    ).to(device)
    
    trainer = TrainingModule(model, device=device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # Accumulation settings
    accumulation_steps = 2
    actual_batch_size = 2
    effective_batch_size = actual_batch_size * accumulation_steps  # 4
    
    model.train()
    accumulated_loss = 0.0
    
    for step in range(accumulation_steps):
        # Mini-batch
        conditions = torch.randn(actual_batch_size, 9, device=device)
        target = torch.randn(actual_batch_size, 3, 512, 512, device=device)
        
        losses = trainer(conditions, target)
        loss = losses['total'] / accumulation_steps
        loss.backward()
        
        accumulated_loss += loss.item()
    
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"Actual batch size: {actual_batch_size}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Accumulated loss: {accumulated_loss:.4f}")


# ============================================================
# EXAMPLE 7: Model Evaluation
# ============================================================
def example_evaluation():
    """Evaluate model and get parameter statistics"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Model Evaluation")
    print("="*60)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = IntegratedConditionalGenerator(
        num_conditions=9,
        initial_image=True,
        device=device
    ).to(device)
    
    # Get parameter counts
    param_counts = model.get_num_parameters()
    print("Parameter Statistics:")
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Trainable parameters: {param_counts['trainable']:,}")
    print(f"  Frozen parameters: {param_counts['frozen']:,}")
    print(f"  Trainable %: {100*param_counts['trainable']/param_counts['total']:.1f}%")
    
    # Model configuration
    print(f"\nModel Configuration:")
    print(f"  Output size: 512x512")
    print(f"  Conditions: {model.num_conditions}")
    print(f"  Noise dim: {model.noise_dim}")
    print(f"  Use VAE: {model.use_vae}")
    print(f"  Initial image support: {model.initial_image}")
    
    # Inference time estimate (rough)
    batch_size = 1
    conditions = torch.randn(batch_size, model.num_conditions, device=device)
    
    model.eval()
    with torch.no_grad():
        import time
        start = time.time()
        for _ in range(10):
            _ = model(conditions)
        elapsed = time.time() - start
        
    print(f"\nInference (10 iterations, batch size 1):")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Per sample: {elapsed/10*1000:.1f}ms")


# ============================================================
# EXAMPLE 8: Checkpoint Loading and Saving
# ============================================================
def example_checkpoints():
    """Save and load model checkpoints"""
    print("\n" + "="*60)
    print("EXAMPLE 8: Checkpoint Management")
    print("="*60)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = IntegratedConditionalGenerator(
        num_conditions=9,
        device=device
    ).to(device)
    
    # Save checkpoint
    checkpoint_path = 'model_checkpoint.pt'
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
    
    # Load checkpoint
    new_model = IntegratedConditionalGenerator(
        num_conditions=9,
        device=device
    ).to(device)
    new_model.load_state_dict(torch.load(checkpoint_path))
    print(f"Model loaded from {checkpoint_path}")
    
    # Full checkpoint with training state
    full_checkpoint = {
        'model': model.state_dict(),
        'epoch': 50,
        'optimizer': optim.Adam(model.parameters()).state_dict(),
    }
    torch.save(full_checkpoint, 'full_checkpoint.pt')
    print("Full checkpoint with training state saved!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("IntegratedConditionalGenerator - Examples")
    print("="*60)
    
    # Run examples
    try:
        example_minimal()
    except Exception as e:
        print(f"Example 1 error: {e}")
    
    try:
        example_training()
    except Exception as e:
        print(f"Example 2 error: {e}")
    
    try:
        example_image_to_image()
    except Exception as e:
        print(f"Example 3 error: {e}")
    
    try:
        example_fixed_noise()
    except Exception as e:
        print(f"Example 4 error: {e}")
    
    try:
        example_vae()
    except Exception as e:
        print(f"Example 5 error: {e}")
    
    try:
        example_gradient_accumulation()
    except Exception as e:
        print(f"Example 6 error: {e}")
    
    try:
        example_evaluation()
    except Exception as e:
        print(f"Example 7 error: {e}")
    
    try:
        example_checkpoints()
    except Exception as e:
        print(f"Example 8 error: {e}")
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
