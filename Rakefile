require 'rake'
require_relative './third/tensorflow-rakefile/tfrake.rb'
include TFRake


define_tasks('font2char_lm', define_pytest: false)


task_in_venv :pytest do
  Dir.glob('font2char_lm/**/*_test.py').each do |file|
    vsh :pytest, file
  end
end


task :examples

%w(char_lm).each do |dir|
  name = "#{dir}_example"

  task_in_venv name do
    vsh "cd #{File.join 'examples', dir} && CUDA_VISIBLE_DEVICES=0 rake"
  end

  Rake::Task[:examples].enhance [name]
end


task :test => %i(pytest examples)
