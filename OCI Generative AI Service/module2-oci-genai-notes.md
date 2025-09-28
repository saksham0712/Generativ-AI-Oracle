# Complete Notes: Oracle Generative AI Certification - Module 2
# OCI Generative AI Service
*Enhanced with Visual Diagrams and Practical Examples*

## Module Overview

Module 2 focuses on Oracle Cloud Infrastructure (OCI) Generative AI Service - a fully managed, serverless platform that provides customizable large language models through a single API. This module covers practical implementation, fine-tuning, security, and hands-on deployment of AI models in production environments.

### Learning Objectives
- Understanding OCI Generative AI Service architecture and components
- Learning to implement and configure chat and embedding models
- Mastering fine-tuning techniques (T-Few and LoRA)
- Implementing dedicated AI clusters for security and performance
- Deploying and managing custom models in production
- Understanding pricing, sizing, and optimization strategies

---

## Lesson 1: OCI Generative AI Service Introduction

### Service Overview

![OCI Generative AI Architecture](generated_image:16)

**OCI Generative AI Service** is a fully managed service providing customizable large language models via a single API for building generative AI applications.

### Key Characteristics

#### **1. Single API Access**
- **Flexibility**: Use different foundational models with minimal code changes
- **Consistency**: Unified interface across all model types
- **Simplicity**: Reduce integration complexity

#### **2. Serverless Architecture**
- **No Infrastructure Management**: Oracle handles all backend operations
- **Automatic Scaling**: Resources scale based on demand
- **Pay-per-Use**: Cost optimization through usage-based pricing

#### **3. Three Core Components**

**Pre-trained Foundational Models**:
- Choice of models from Meta and Cohere
- Chat models for conversational AI
- Embedding models for semantic search

**Flexible Fine-tuning**:
- Create custom models with your own datasets
- Parameter-efficient methods (T-Few, LoRA)
- Domain-specific optimization

**Dedicated AI Clusters**:
- GPU-based compute resources
- Isolated customer environments
- Ultra-low latency RDMA networking

### How It Works

**Input → Processing → Output**:
1. **Text Input**: Provide prompts in natural language
2. **Processing**: AI service reasons over text and context
3. **Intelligent Response**: Generate contextually appropriate answers

**Use Cases Enabled**:
- **Chat**: Conversational interfaces and dialogue systems
- **Text Generation**: Content creation and completion
- **Information Retrieval**: Document search and Q&A
- **Semantic Search**: Meaning-based content discovery

---

## Lesson 2: Pre-trained Foundational Models

### Chat Models

#### **Command-R-Plus**
- **Provider**: Cohere family of LLMs
- **Capability**: Highly performant instruction-following conversational model
- **User Prompt Limit**: Up to 128,000 tokens
- **Response Limit**: Up to 4,000 tokens per run
- **Use Cases**: Advanced applications, complex reasoning, large document processing
- **Cost**: Higher but more powerful

#### **Command-R (16K)**
- **Provider**: Cohere family
- **Capability**: Smaller, faster version of Command-R-Plus
- **User Prompt Limit**: 16,000 tokens
- **Response Limit**: 4,000 tokens per run
- **Use Cases**: Entry-level applications where speed and cost matter
- **Cost**: More affordable alternative

#### **Llama 3.1 Family (Meta)**
- **Models**: 
  - 405 billion parameter model (largest publicly available)
  - 70 billion parameter model
- **Prompt/Response Limit**: Up to 128,000 tokens each
- **Capability**: Complex enterprise-level applications
- **Provider**: Meta

### Chat Model Features

#### **Conversational Context**
- **Memory**: Models keep context of previous prompts
- **Follow-up**: Continue conversations with related questions
- **Coherence**: Maintain conversation flow and relevance

#### **Instruction-Following**
- **Instruction Tuning**: Additional training on human language instructions
- **Task Execution**: "Generate an email," "Summarize this text"
- **Better Compliance**: Improved adherence to user directions

### Embedding Models

#### **Understanding Embeddings**
**Definition**: Text converted to vectors of numbers that capture semantic meaning

**Purpose**: Enable computers to understand relationships between text pieces

#### **Available Models**

**Embed English Model**:
- **Language**: English language optimization
- **Use Case**: English-only semantic search applications

**Embed Multilingual Model**:
- **Languages**: 100+ language support
- **Capabilities**:
  - Within-language search: French query on French documents
  - Cross-language search: Chinese query on French documents
- **Flexibility**: Global application deployment

#### **Use Cases for Embeddings**
- **Semantic Search**: Meaning-based rather than keyword-based search
- **Document Similarity**: Find related content
- **Recommendation Systems**: Content suggestions
- **Classification**: Automated content categorization

---

## Lesson 3: Tokens and Model Parameters

### Understanding Tokens

#### **Token Basics**
- **LLM Understanding**: Models process tokens, not characters
- **Token Types**:
  - Part of a word: "friend" + "ship" = friendship
  - Entire word: "apple" = single token
  - Punctuation: "," "." "!" = individual tokens

#### **Token Estimation Guidelines**
- **Simple Text**: ~1 token per word average
- **Complex Text**: 2-3 tokens per word average (technical terms, uncommon words)

#### **Tokenization Example**
```
Sentence: "Many words map to one token, but some don't, indivisible."
Tokens: ["Many", "words", "map", "to", "one", "token", ",", "but", "some", "don", "'t", ",", "in", "div", "isible", "."]
Word Count: 10 words
Token Count: 15 tokens
```

### Chat Model Parameters

![Temperature Parameter Effects](generated_image:21)

#### **Maximum Output Tokens**
- **Definition**: Maximum number of tokens model generates per response
- **Impact**: Controls response length and computational cost
- **Strategy**: Set based on use case requirements

#### **Preamble Override**
- **Purpose**: Initial guideline message that changes model behavior
- **Default**: Each model has standard preamble
- **Customization**: Override with specific style/tone requirements

**Example**:
```
Default Preamble: Standard professional response
Custom Preamble: "Answer in a pirate tone"
Result: "Ahoy! Here be the answer ye seek, matey!"
```

#### **Temperature Control**
- **Range**: 0 (deterministic) to higher values (creative)
- **Low Temperature (0.1)**: 
  - Consistent, predictable outputs
  - Factual question answering
  - Professional applications
- **High Temperature (1.5+)**:
  - Creative, varied outputs
  - Story generation
  - Brainstorming applications

#### **Advanced Parameters**

**Top-K Sampling**:
- **Definition**: Pick next token from top K tokens by probability
- **Example**: K=3 limits selection to 3 highest probability tokens
- **Use Case**: Control randomness while maintaining quality

**Top-P (Nucleus Sampling)**:
- **Definition**: Select tokens until cumulative probability reaches P
- **Example**: P=0.9 uses tokens covering 90% of probability mass
- **Benefit**: Dynamic vocabulary size based on confidence

**Frequency & Presence Penalties**:
- **Frequency Penalty**: Reduces repetition based on occurrence count
- **Presence Penalty**: Reduces repetition regardless of frequency
- **Goal**: More natural, less repetitive text generation

---

## Lesson 4: Fine-tuning and Custom Models

### Fine-tuning Overview

![Fine-tuning Workflow](chart:17)

**Definition**: Optimizing a pre-trained foundational model on smaller, domain-specific datasets to create custom models tailored for specific tasks.

### When to Use Fine-tuning

#### **Scenarios**
- Pre-trained model doesn't perform well on your specific task
- Need to teach the model domain-specific knowledge
- Require consistent style/format in outputs
- Want to improve performance on specialized tasks

#### **Benefits**
- **Improved Performance**: Better accuracy on specific domains
- **Model Efficiency**: Optimized for particular use cases
- **Consistency**: Predictable outputs for business applications

### T-Few Fine-tuning

![T-Few Process](generated_image:20)

#### **Technology Overview**
- **Method**: Parameter Efficient Fine-Tuning (PEFT)
- **Approach**: Insert new layers (0.01% of base model size)
- **Efficiency**: Update only fraction of model weights
- **Result**: Reduced training time and cost vs. vanilla fine-tuning

#### **T-Few Process**
1. **Base Model**: Start with pre-trained foundational model
2. **Additional Layers**: Insert small supplementary layers
3. **Selective Training**: Update only T-Few layers during training
4. **Weight Isolation**: Localize changes to new layers only
5. **Custom Model**: Deploy fine-tuned model with enhanced capabilities

#### **Advantages**
- **Cost Effective**: Significantly cheaper than full fine-tuning
- **Faster Training**: Reduced computational requirements
- **Memory Efficient**: Shared base model weights across fine-tuned versions
- **Quality Results**: Comparable performance to full fine-tuning

### LoRA (Low Rank Adaptation)

#### **Alternative PEFT Method**
- **Approach**: Add special "gears" to adjust model behavior
- **Implementation**: Insert low-rank matrices into existing layers
- **Flexibility**: Another parameter-efficient option

### Training Data Requirements

#### **JSONL Format**
**Required Structure**:
```json
{"prompt": "Human request or input", "completion": "Expected AI response"}
{"prompt": "Ask my aunt about the walk", "completion": "Can you go to the walk with me?"}
```

**Format Requirements**:
- **File Type**: JSONL (JSON Lines)
- **Encoding**: UTF-8
- **Structure**: Each line contains one training example
- **Properties**: Must have "prompt" and "completion" fields

#### **Data Preparation Example**
**Original Data**: Conversational request → AI assistant response
**Processed Format**: 
- Human request becomes "prompt"
- AI assistant utterance becomes "completion"
- 2,000+ examples for training

---

## Lesson 5: Dedicated AI Clusters

### Architecture and Security

![Dedicated AI Clusters](generated_image:18)

#### **Core Concept**
**Dedicated AI Clusters**: GPU-based compute resources that host customer fine-tuning and inference workloads in isolated environments.

#### **Key Components**
- **Dedicated GPUs**: Exclusive allocation per customer
- **RDMA Networking**: Ultra-low latency cluster networking
- **Security Isolation**: Customer data/models separated from others
- **Scalable Resources**: Create large GPU clusters as needed

### Cluster Types

![Cluster Types Comparison](chart:19)

#### **1. Small Cohere Dedicated**
- **Models Supported**: Command R Light, Command R 08-2024
- **Use Cases**: Fine-tuning and hosting Cohere models
- **Requirements**: 
  - Fine-tuning: 2-8 units (varies by model)
  - Hosting: 1+ units

#### **2. Large Cohere Dedicated**
- **Models Supported**: Command R Plus, Command R family
- **Use Cases**: Advanced Cohere model applications
- **Requirements**: Higher resource allocation for larger models

#### **3. Embed Cohere Dedicated**
- **Models Supported**: Embed English, Embed Multilingual
- **Use Cases**: Hosting embedding models only
- **Note**: No fine-tuning support (embedding models aren't fine-tuned)
- **Requirements**: 1 unit for hosting

#### **4. Large Meta Dedicated**
- **Models Supported**: 
  - Llama 3.1 (70B, 405B parameters)
  - Llama 3.2 Vision (11B, 90B parameters)
- **Requirements**:
  - Fine-tuning: 4 units
  - Hosting: 1 unit

### Cluster Usage Patterns

#### **Fine-tuning Clusters**
- **Purpose**: Training custom models
- **Duration**: Hours to complete (not 24/7)
- **Billing**: Hourly usage (minimum 1 hour)
- **Efficiency**: Create, train, terminate

#### **Hosting Clusters**
- **Purpose**: Serving inference traffic
- **Duration**: Monthly commitment (744 hours)
- **Capacity**: Host multiple models simultaneously
- **Example**: Up to 50 models with T-Few methodology

### Memory and Performance Optimization

#### **Parameter Sharing**
- **Base Model**: Loaded once in GPU memory
- **Custom Models**: Only delta weights (0.01% additional)
- **Efficiency**: Minimal memory overhead when switching models
- **Result**: Cost-effective multi-model serving

#### **Concurrent Serving**
- **Multiple Endpoints**: Base + custom models serve simultaneously
- **Resource Sharing**: Efficient GPU utilization
- **Low Latency**: Fast model switching due to shared weights

---

## Lesson 6: Security and Privacy

### Security Architecture

#### **Core Design Principles**
- **Customer Isolation**: GPUs allocated exclusively per customer
- **Data Privacy**: No cross-customer data access
- **Network Security**: Dedicated RDMA clusters
- **Model Isolation**: Custom models confined to customer tenancy

#### **Multi-layered Security**

**1. Infrastructure Level**:
- **Dedicated GPUs**: Isolated compute resources
- **RDMA Networking**: Exclusive network fabric
- **Physical Separation**: Hardware-level isolation

**2. Data Level**:
- **Encryption**: Data encrypted at rest and in transit
- **Access Control**: Customer-only access to models/data
- **Audit Trails**: Comprehensive logging and monitoring

**3. Model Level**:
- **Tenancy Isolation**: Models accessible only within customer tenancy
- **Endpoint Security**: Authenticated and authorized access only
- **Version Control**: Secure model versioning and deployment

### Integration with OCI Security Services

#### **Identity and Access Management (IAM)**
- **Authentication**: User and service authentication
- **Authorization**: Fine-grained access control
- **Example**: Application X accesses Custom Model X, Application Y accesses Base Model

#### **Key Management Service**
- **Encryption Keys**: Secure storage of model encryption keys
- **Key Rotation**: Automated key lifecycle management
- **Compliance**: Meet regulatory encryption requirements

#### **Object Storage Security**
- **Model Weights**: Encrypted storage of fine-tuned model weights
- **Access Policies**: Controlled access to training data and models
- **Backup/Recovery**: Secure model backup and restoration

---

## Lesson 7: Practical Implementation and APIs

### OCI Console Playground

#### **Visual Interface**
- **No-Code Testing**: Explore models without programming
- **Parameter Adjustment**: Real-time parameter tuning
- **Code Generation**: Automatic SDK code generation
- **Integration**: Copy-paste code into applications

#### **Supported Languages**
- **Python SDK**: Complete API coverage
- **Java SDK**: Enterprise integration support
- **REST APIs**: Language-agnostic HTTP interfaces

### Python SDK Implementation

#### **Basic Setup**
```python
import oci

# Authentication Configuration
compartment_id = "your-compartment-id"
config_profile = "DEFAULT"
config = oci.config.from_file("~/.oci/config", config_profile)

# Service Endpoint
service_endpoint = "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com"

# Create Client
genai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
    config=config,
    service_endpoint=service_endpoint,
    retry_strategy=oci.retry.NoneRetryStrategy(),
    timeout=(10, 240)
)
```

#### **Chat Request Example**
```python
# Define Chat Request
chat_request = oci.generative_ai_inference.models.CohereChatRequest(
    message="Generate a job description for a data visualization expert",
    max_tokens=600,
    temperature=0,
    is_stream=False
)

# Create Chat Details
chat_detail = oci.generative_ai_inference.models.ChatDetails(
    serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
        model_id="cohere.command-r-plus"
    ),
    chat_request=chat_request,
    compartment_id=compartment_id
)

# Send Request
chat_response = genai_inference_client.chat(chat_detail)
```

### Embedding Implementation

#### **Semantic Search Setup**
```python
# Embedding Request
embed_text_details = oci.generative_ai_inference.models.EmbedTextDetails(
    inputs=["Article about technical skills", "Career development guide"],
    serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
        model_id="cohere.embed-english-v3.0"
    ),
    compartment_id=compartment_id
)

# Get Embeddings
embed_response = genai_inference_client.embed_text(embed_text_details)
```

---

## Lesson 8: Pricing and Cost Optimization

### Service Limit Management

#### **Default State**
- **Initial Limits**: All dedicated AI cluster limits set to zero
- **Activation Required**: Submit service limit increase requests
- **SKU Names**: 
  - `dedicated-unit-small-cohere-count`
  - `dedicated-unit-large-cohere-count`
  - `dedicated-unit-embed-cohere-count`
  - `dedicated-unit-large-meta-count`

### Pricing Model

#### **Fine-tuning Costs**
- **Billing**: Hourly usage during training
- **Minimum**: 1 hour (even if training takes 30 minutes)
- **Duration**: Actual training time (typically hours, not days)

#### **Hosting Costs**
- **Commitment**: Monthly (744 hours)
- **No Partial Hosting**: Must commit to full month
- **Multi-model**: Host up to 50 models on same cluster (T-Few)

### Cost Optimization Example

#### **Scenario**: Weekly fine-tuning + monthly hosting
- **Model**: Command R 08-2024
- **Fine-tuning**: 8 small Cohere units × 5 hours × 4 weeks = 160 unit-hours/month
- **Hosting**: 1 small Cohere unit × 744 hours = 744 unit-hours/month
- **Total**: 904 unit-hours/month × $6.50/hour = ~$5,900/month

#### **Optimization Strategies**
- **Batch Training**: Multiple models in single session
- **Shared Hosting**: Multiple models on same cluster
- **Right-sizing**: Choose appropriate cluster size
- **Regional Pricing**: Compare costs across regions

---

## Lesson 9: Hands-on Implementation Guide

### Step 1: Environment Setup

#### **Prerequisites**
1. **OCI Account**: Active Oracle Cloud account
2. **Region Selection**: Choose region with Generative AI availability
3. **Service Limits**: Request necessary dedicated cluster limits
4. **IAM Policies**: Configure appropriate access permissions

### Step 2: Create Dedicated AI Clusters

#### **Fine-tuning Cluster Creation**
```bash
# Via OCI Console:
# Analytics & AI → Generative AI → Dedicated AI Clusters → Create
Name: custom-fine-tuning-cluster
Type: Fine-tuning
Base Model: Command Light (requires small Cohere units)
Commitment: Hourly usage
```

#### **Hosting Cluster Creation**
```bash
Name: hosting-cluster
Type: Hosting
Base Model: Command Light
Commitment: 744 unit hours (monthly)
Capacity: 50 models (T-Few methodology)
```

### Step 3: Prepare Training Data

#### **Data Format Requirements**
```json
{"prompt": "Human request", "completion": "AI response"}
{"prompt": "Ask my aunt if she can go to the JDRF walk", "completion": "Can you go to the JDRF walk with me?"}
```

#### **Data Upload**
1. **Format**: Convert data to JSONL
2. **Upload**: Store in OCI Object Storage
3. **Permissions**: Configure IAM policies for service access
4. **Validation**: Ensure UTF-8 encoding and proper structure

### Step 4: Fine-tuning Process

#### **Model Creation Workflow**
1. **Navigate**: Custom Models → Create Model
2. **Configure**: 
   - Base Model: Match to cluster unit type
   - Fine-tuning Method: T-Few (recommended)
   - Dedicated Cluster: Select fine-tuning cluster
3. **Hyperparameters**: Use defaults initially
4. **Training Data**: Select uploaded JSONL file
5. **Submit**: Start fine-tuning process

#### **Monitor Progress**
- **Training Time**: Typically 5-30 minutes depending on data size
- **Metrics**: Monitor accuracy and loss during training
- **Completion**: Model status changes to "Active"

### Step 5: Model Evaluation

#### **Accuracy Metrics**
- **Accuracy**: Percentage of output tokens matching training data
- **Target**: 0.9+ (90%+ accuracy) considered excellent
- **Loss**: Should trend toward 0 (perfect outputs)

#### **Qualitative Testing**
```python
# Test with unseen data
test_prompt = "Turn this message into a virtual assistant action"
# Compare base model vs custom model responses
# Evaluate consistency across different temperature settings
```

### Step 6: Create and Deploy Endpoint

#### **Endpoint Configuration**
1. **Navigate**: Endpoints → Create Endpoint
2. **Model Selection**: Choose custom model
3. **Hosting Cluster**: Select hosting cluster
4. **Content Moderation**: Enable if needed for production
5. **Deploy**: Endpoint becomes active in minutes

#### **Production Testing**
```python
# Test endpoint with production-like traffic
# Verify response quality and consistency
# Monitor performance metrics
```

---

## Lesson 10: Best Practices and Optimization

### Fine-tuning Best Practices

#### **Data Quality**
- **Volume**: 1,000+ examples for good results
- **Diversity**: Varied examples covering use cases
- **Quality**: Clean, consistent formatting
- **Balance**: Avoid bias in training examples

#### **Hyperparameter Tuning**
- **Start Simple**: Use default values initially
- **Iterative Improvement**: Adjust based on metrics
- **Learning Rate**: Critical for convergence
- **Batch Size**: Balance memory and training stability

### Production Deployment

#### **Performance Optimization**
- **Right-sizing**: Choose appropriate cluster size
- **Multi-model**: Leverage shared hosting for cost efficiency
- **Caching**: Implement response caching where appropriate
- **Monitoring**: Track usage patterns and performance

#### **Security Best Practices**
- **Access Control**: Implement least-privilege IAM policies
- **Network Security**: Use private endpoints where possible
- **Data Governance**: Implement data lifecycle policies
- **Audit Logging**: Enable comprehensive audit trails

### Cost Management

#### **Usage Optimization**
- **Scheduled Training**: Batch fine-tuning jobs
- **Model Lifecycle**: Deactivate unused endpoints
- **Resource Planning**: Forecast usage patterns
- **Cost Monitoring**: Set up budget alerts

#### **Architecture Patterns**
- **Hybrid Approach**: Combine prompting, fine-tuning, and RAG
- **Progressive Enhancement**: Start simple, add complexity as needed
- **A/B Testing**: Compare model performance and costs

---

## Key Takeaways and Exam Focus

### **Service Architecture Understanding**
- OCI Generative AI Service provides fully managed LLM capabilities
- Three core components: pre-trained models, fine-tuning, dedicated clusters
- Serverless architecture eliminates infrastructure management

### **Model Types and Use Cases**
- Chat models (Command-R family, Llama 3.1) for conversational AI
- Embedding models for semantic search and similarity
- Choose models based on capability requirements and cost constraints

### **Fine-tuning Implementation**
- T-Few methodology for parameter-efficient customization
- JSONL format requirements for training data
- Dedicated clusters provide secure, isolated training environments

### **Security and Isolation**
- Customer data and models completely isolated
- Integration with OCI security services (IAM, KMS, Object Storage)
- Enterprise-grade security for production deployments

### **Practical Implementation**
- OCI Console Playground for no-code testing and development
- Python/Java SDKs for programmatic integration
- REST APIs for platform-agnostic implementations

### **Cost Optimization Strategies**
- Understanding service limits and pricing models
- Efficient resource allocation and usage patterns
- Multi-model hosting for cost efficiency

### **Production Considerations**
- Proper data preparation and quality assurance
- Performance monitoring and optimization
- Security best practices and compliance requirements

---

## Additional Resources for Deep Dive

### **Oracle Documentation**
- OCI Generative AI Service User Guide
- API Reference Documentation
- Security Best Practices Guide
- Pricing and Service Limits

### **Hands-on Practice**
- OCI Console Playground exploration
- SDK integration exercises
- Fine-tuning with sample datasets
- Endpoint deployment and testing

### **Advanced Topics**
- Multi-modal model capabilities
- Advanced prompting strategies
- RAG integration patterns
- Performance optimization techniques

This concludes the comprehensive coverage of Oracle's Generative AI Certification Module 2 on OCI Generative AI Service. The module provides both theoretical understanding and practical implementation skills necessary for deploying enterprise-grade generative AI solutions on Oracle Cloud Infrastructure.
