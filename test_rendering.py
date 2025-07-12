import glfw

if not glfw.init():
    raise Exception("GLFW initialization failed")

# Then set hints if needed
# glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
# glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
# glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
# glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

# Create window
window = glfw.create_window(800, 600, "OpenGL Test", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window creation failed")

glfw.make_context_current(window)
# Now you can call gladLoadGL, render, etc.
