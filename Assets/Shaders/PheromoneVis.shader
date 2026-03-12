Shader "Simulation/PheromoneVis"
{
    Properties
    {
        _PheromoneTex ("Pheromone Texture", 2D) = "black" {}
        _FoodColor ("Food Trail Color", Color) = (1, 0.3, 0, 1)
        _HomeColor ("Home Trail Color", Color) = (0, 0.7, 1, 1)
        _Brightness ("Brightness", Float) = 3.0
    }
    SubShader
    {
        Tags
        {
            "RenderPipeline" = "UniversalPipeline"
            "RenderType" = "Transparent"
            "Queue" = "Geometry-1"
        }

        Pass
        {
            Name "PheromoneVis"
            Tags { "LightMode" = "UniversalForward" }

            ZWrite Off
            Blend One One

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            TEXTURE2D(_PheromoneTex);
            SAMPLER(sampler_PheromoneTex);

            CBUFFER_START(UnityPerMaterial)
                float4 _PheromoneTex_ST;
                float4 _FoodColor;
                float4 _HomeColor;
                float _Brightness;
            CBUFFER_END

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                OUT.positionHCS = TransformObjectToHClip(IN.positionOS.xyz);
                OUT.uv = TRANSFORM_TEX(IN.uv, _PheromoneTex);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                float4 pheromone = SAMPLE_TEXTURE2D(_PheromoneTex, sampler_PheromoneTex, IN.uv);
                float food = saturate(pheromone.r * _Brightness);
                float home = saturate(pheromone.g * _Brightness);
                float3 col = _FoodColor.rgb * food + _HomeColor.rgb * home;
                return half4(col, 1.0);
            }
            ENDHLSL
        }
    }
}
