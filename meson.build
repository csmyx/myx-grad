project('myx_grad ', 'cpp')

inc = include_directories('include/')
fmt_dep = dependency('fmt')
catch2_dep = dependency('catch2-with-main')

executable(
    'myx_grad',
    'src/myx_grad.cpp',
    'src/main.cpp',
    include_directories: inc,
    dependencies: [fmt_dep],
    install: true,
)

# add test
test(
    'myx_grad_test',
    executable(
        'test1',
        'tests/test1.cpp',
        'src/arena.cpp',
        include_directories: inc,
        dependencies: [fmt_dep, catch2_dep]
    )
)