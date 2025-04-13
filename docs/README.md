# RNA Predict Documentation

This directory contains comprehensive documentation for the RNA Predict project.

## Directory Structure

### `/components`
Documentation for individual components of the pipeline:
- `/stageA` - 2D Adjacency prediction
- `/stageB` - Torsion and pairwise embeddings
- `/stageC` - 3D Reconstruction
- `/stageD` - Diffusion refinement
- `/unified_latent` - Latent merger components

### `/integration`
Documentation for integrating components:
- `/hydra` - Hydra configuration and integration
- `/pipeline` - Full pipeline orchestration
- `/testing` - Integration testing

### `/reference`
Reference documentation:
- `/api` - API documentation
- `/architecture` - Architecture diagrams and explanations
- `/methods` - Detailed explanations of methods used

### `/guides`
User and developer guides:
- `/getting_started` - Getting started guides
- `/best_practices` - Best practices for development
- `/debugging` - Debugging guides

### `/advanced`
Advanced topics:
- `/perceiver_io` - Perceiver IO implementation details
- `/energy_minimization` - Energy minimization and MD
- `/optimization` - Performance optimization

## Documentation Standards

All documentation should follow these standards:
1. Use Markdown format
2. Include a clear title and purpose
3. Provide code examples where appropriate
4. Include references to related documentation
5. Keep files focused on a single topic
6. Use consistent terminology across documents

## Contributing to Documentation

When adding new documentation:
1. Place it in the appropriate subdirectory
2. Update this README if adding new directories
3. Link to it from related documentation
4. Follow the documentation standards 