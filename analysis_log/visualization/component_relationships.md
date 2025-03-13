# Component Relationship Visualization

**Tags:** #architecture #integration #visualization #summary

## Overview
This document provides visual representations of the relationships between various components of the CS2 reinforcement learning agent system using Mermaid diagrams.

## High-Level System Architecture

```mermaid
graph TD
    subgraph "Game Interface"
        GI[Bridge Mod]
        CV[Computer Vision]
    end
    
    subgraph "Agent System"
        SA[Strategic Agent]
        AA[Adaptive Agent]
        RM[Reward Mechanism]
    end
    
    subgraph "Action System"
        AM[Action Mapper]
        AP[Action Processor]
    end
    
    subgraph "Vision System"
        VI[Vision Interface]
        OV[Ollama Vision]
        FE[Feature Extraction]
    end
    
    subgraph "Infrastructure"
        CS[Configuration System]
        ER[Error Recovery]
        PP[Parallel Processing]
    end

    CV --> VI
    VI --> OV
    OV --> FE
    FE --> SA
    FE --> RM
    
    SA --> AA
    AA --> AM
    AM --> AP
    AP --> GI

    RM --> SA
    CS --> GI
    CS --> VI
    CS --> SA
    
    ER --> VI
    ER --> AM
    ER --> GI
    
    PP --> VI
    PP --> OV
```

## Data Flow Diagram

```mermaid
flowchart LR
    GI[Game Interface] --> |Raw Screenshots| VI[Vision Interface]
    VI --> |Processed Images| OV[Ollama Vision]
    OV --> |Scene Understanding| FE[Feature Extraction]
    FE --> |Game State Features| RM[Reward Mechanism]
    FE --> |Game State Features| SA[Strategic Agent]
    RM --> |Reward Signals| SA
    SA --> |Strategic Decisions| AA[Adaptive Agent]
    AA --> |Mode-specific Actions| AM[Action Mapper]
    AM --> |Game Commands| AP[Action Processor]
    AP --> |Executable Actions| GI
    
    style GI fill:#f9d,stroke:#333,stroke-width:2px
    style VI fill:#bbf,stroke:#333,stroke-width:2px
    style OV fill:#bbf,stroke:#333,stroke-width:2px
    style FE fill:#bbf,stroke:#333,stroke-width:2px
    style RM fill:#bfb,stroke:#333,stroke-width:2px
    style SA fill:#bfb,stroke:#333,stroke-width:2px
    style AA fill:#bfb,stroke:#333,stroke-width:2px
    style AM fill:#fbb,stroke:#333,stroke-width:2px
    style AP fill:#fbb,stroke:#333,stroke-width:2px
```

## Component Dependencies

```mermaid
graph TD
    subgraph "Core Components"
        SA[Strategic Agent]
        AA[Adaptive Agent]
        VI[Vision Interface]
        AM[Action Mapper]
    end
    
    subgraph "Supporting Components"
        CS[Configuration System]
        ER[Error Recovery]
        PP[Parallel Processing]
        RM[Reward Mechanism]
    end
    
    subgraph "Integration Points"
        GI[Game Interface]
        FE[Feature Extraction]
        OV[Ollama Vision]
        AP[Action Processor]
    end
    
    SA --> AA
    AA --> AM
    VI --> OV
    OV --> FE
    AM --> AP
    
    FE --> SA
    FE --> RM
    RM --> SA
    
    CS -.-> SA
    CS -.-> VI
    CS -.-> GI
    CS -.-> AM
    
    ER -.-> VI
    ER -.-> AM
    ER -.-> GI
    
    PP -.-> VI
    PP -.-> OV
    
    AP -.-> GI
    GI -.-> VI
    
    classDef core fill:#f96,stroke:#333,stroke-width:2px
    classDef support fill:#9cf,stroke:#333,stroke-width:2px
    classDef integration fill:#c9f,stroke:#333,stroke-width:2px
    
    class SA,AA,VI,AM core
    class CS,ER,PP,RM support
    class GI,FE,OV,AP integration
```

## Performance Bottlenecks

```mermaid
graph LR
    GI[Game Interface] --> VI[Vision Interface]
    VI --> OV[Ollama Vision]
    OV --> FE[Feature Extraction]
    FE --> SA[Strategic Agent]
    SA --> AA[Adaptive Agent]
    AA --> AM[Action Mapper]
    AM --> AP[Action Processor]
    AP --> GI
    
    classDef bottleneck fill:#f55,stroke:#333,stroke-width:4px
    classDef minor fill:#fa5,stroke:#333,stroke-width:2px
    classDef normal fill:#5d5,stroke:#333,stroke-width:1px
    
    class VI,OV bottleneck
    class SA,AM minor
    class GI,FE,AA,AP normal
```

## Testing Coverage

```mermaid
graph TD
    subgraph "Components"
        GI[Game Interface]
        VI[Vision Interface]
        OV[Ollama Vision]
        FE[Feature Extraction]
        SA[Strategic Agent]
        AA[Adaptive Agent]
        AM[Action Mapper]
        AP[Action Processor]
        RM[Reward Mechanism]
    end
    
    subgraph "Test Types"
        UT[Unit Tests]
        IT[Integration Tests]
        PT[Performance Tests]
        ST[Simulation Tests]
    end
    
    UT --> GI
    UT --> VI
    UT --> SA
    UT --> AA
    UT --> AM
    UT --> AP
    UT --> RM
    
    IT --> VI
    IT --> OV
    IT --> FE
    IT --> SA
    IT --> AM
    IT --> AP
    
    PT --> VI
    PT --> OV
    PT --> SA
    
    ST --> SA
    ST --> AA
    ST --> AM
    ST --> RM
    
    classDef highCoverage fill:#5d5,stroke:#333,stroke-width:2px
    classDef mediumCoverage fill:#fd5,stroke:#333,stroke-width:2px
    classDef lowCoverage fill:#f55,stroke:#333,stroke-width:2px
    
    class GI,VI,SA,AM,RM highCoverage
    class AA,AP,FE mediumCoverage
    class OV lowCoverage
```

## How to Use These Diagrams

The diagrams above provide different views of the system:

1. **High-Level System Architecture** - Shows the major subsystems and their relationships
2. **Data Flow Diagram** - Illustrates how data flows through the system from game interface to actions
3. **Component Dependencies** - Highlights which components depend on others
4. **Performance Bottlenecks** - Identifies the performance-critical components
5. **Testing Coverage** - Shows the test coverage across different components

These visualizations complement the written analyses and provide a quick understanding of system structure and relationships.

## Related Documents
- [Comprehensive Architecture](../architecture/comprehensive_architecture.md)
- [Component Integration](../architecture/component_integration.md)
- [Performance Profiling Overview](../performance/performance_profiling.md)
- [Testing Infrastructure](../testing/testing_infrastructure.md) 