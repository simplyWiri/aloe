#include <aloe/core/CommandList.h>
#include <aloe/core/Device.h>
#include <aloe/core/FrameGraph.h>
#include <aloe/core/PipelineManager.h>
#include <aloe/core/ResourceManager.h>
#include <aloe/core/Swapchain.h>
#include <aloe/core/TaskGraph.h>

struct GameOfLifeState {
    // --- Simulation State
    aloe::ImageDesc sim_img = {
        .extent = { 4096, 4096, 1 },
        .format = VK_FORMAT_R8_UINT,
        .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
    };

    aloe::ImageHandle prev_img;
    aloe::ImageHandle next_img;

    aloe::ShaderUniform<aloe::ImageHandle> prev_uni;
    aloe::ShaderUniform<aloe::ImageHandle> next_uni;

    aloe::PipelineHandle sim_pipeline;

    // --- Rendering State
    aloe::ImageHandle render_target;

    aloe::ShaderUniform<aloe::ImageHandle> sim_output_uni;

    aloe::PipelineHandle render_pipeline;

    void allocate_compute_resources( aloe::ResourceManager& rm, aloe::PipelineManager& pm ) {
        // We use ping-pong buffers for simulation resources, so these handles get swapped each frame, hence the (_a, _b)
        // names. `next_img` is used as the seed for the simulation, so upload initial patterns to that image.
        prev_img = rm.declare_image( "gol_sim_a", sim_img );
        next_img = rm.declare_image( "gol_sim_b", sim_img );

        sim_pipeline = pm.compile_pipeline( { .compute_shader = { "game_of_life.slang", "compute_main" } } ).value();

        prev_uni = pm.get_uniform<aloe::ImageHandle>( sim_pipeline, "prev_state" );
        next_uni = pm.get_uniform<aloe::ImageHandle>( sim_pipeline, "next_state" );
    }

    void attach_compute_pass( aloe::TaskGraph& graph ) {
        graph.add_task( {
            .name = "Game of Life Simulation",
            .queue_type = aloe::TaskDesc::Compute,
            .resources = {
                aloe::ResourceUsageDesc::make(prev_img, aloe::ResourceUsageDesc::ComputeStorageReadWrite),
                aloe::ResourceUsageDesc::make(next_img, aloe::ResourceUsageDesc::ComputeStorageReadWrite),
            },
            .execute_fn = [&](aloe::CommandList& cmd) -> void {
                // ping pong the resources, for this simulation step.
                std::swap(prev_img, next_img);

                cmd.bind_pipeline( sim_pipeline )
                    .set_uniform( prev_uni.set_value( prev_img ) )
                    .set_uniform( next_uni.set_value( next_img ) )
                    .dispatch( (sim_img.extent.width + 63) / 64, sim_img.extent.height, sim_img.extent.depth );
            }
        });
    }

    void allocate_graphics_resources( aloe::ResourceManager& rm, aloe::PipelineManager& pm ) {
        render_target = rm.declare_image( "gol_render_target",
                                          {
                                              .extent = { 4096, 4096, 1 },
                                              .format = VK_FORMAT_A2B10G10R10_UNORM_PACK32,
                                              .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                                          } );

        sim_pipeline = pm.compile_pipeline( {
                                                .vertex_shader = { "game_of_life.slang", "vertex_main" },
                                                .fragment_shader = { "game_of_life.slang", "fragment_main" },
                                            } )
                           .value();

        sim_output_uni = pm.get_uniform<aloe::ImageHandle>( sim_pipeline, "simulation_state" );
    }

    void attach_graphics_pass( aloe::FrameGraph& graph ) {
        graph.add_task( {
            .name = "Game of Life Rendering",
            .queue_type = aloe::TaskDesc::Graphics,
            .resources = {
                aloe::ResourceUsageDesc::make(prev_img, aloe::ResourceUsageDesc::FragmentStorageRead),
                aloe::ResourceUsageDesc::make(next_img, aloe::ResourceUsageDesc::FragmentStorageRead),
                aloe::ResourceUsageDesc::make(render_target, aloe::ResourceUsageDesc::ColorAttachmentWrite),
            },
            .execute_fn = [&](aloe::CommandList& cmd) -> void {
                cmd.begin_renderpass( {
                    .colors = {
                        {
                            .image = render_target,
                            .format = VK_FORMAT_A2B10G10R10_UNORM_PACK32,
                        }
                    },
                    .render_area = { { 0, 0 }, { 4096, 4096 } }
                } );

                cmd.bind_pipeline( render_pipeline )
                    .set_uniform( sim_output_uni.set_value( next_img ) )
                    .draw( 3, 1, 0, 0 );

                cmd.end_renderpass();
            }
        } );

        graph.set_output_image( render_target );
    }
};

int main() {
    auto device = aloe::Device( {} );
    auto swapchain = device.make_swapchain( {} );
    auto rm = device.make_resource_manager();
    auto pm = device.make_pipeline_manager( {} );
    auto fg = device.make_frame_graph();

    GameOfLifeState sim_state;

    sim_state.allocate_compute_resources( *rm, *pm );
    sim_state.attach_compute_pass( fg );

    sim_state.allocate_graphics_resources( *rm, *pm );
    sim_state.attach_graphics_pass( fg );

    fg.compile();

    while ( !swapchain->poll_events() ) { fg.execute(); }

    return 1;
}
