//
// Created by Hans Johnson on 2019-02-26.
//

#ifndef PART2_INCLASS_WORK_HW2ITERATIONLOGGER_H
#define PART2_INCLASS_WORK_HW2ITERATIONLOGGER_H

template < typename OptimizerType >
class CommandIterationUpdate : public itk::Command
{
public:
  using Self = CommandIterationUpdate;
  using Superclass = itk::Command;
  using Pointer = itk::SmartPointer<Self>;
  itkNewMacro( Self );
protected:
  CommandIterationUpdate() {};
public:
  using OptimizerPointer = const OptimizerType *;
  void Execute(itk::Object *caller, const itk::EventObject & event) override
  {
    Execute( (const itk::Object *)caller, event);
  }
  void Execute(const itk::Object * object, const itk::EventObject & event) override
  {
    OptimizerPointer optimizer = static_cast< OptimizerPointer >( object );
    if( ! itk::IterationEvent().CheckEvent( &event ) )
    {
      return;
    }
    std::cout << optimizer->GetCurrentIteration() << " = ";
    std::cout << optimizer->GetValue() << " : ";
    std::cout << optimizer->GetCurrentPosition() << std::endl;

  }
};




#endif //PART2_INCLASS_WORK_HW2ITERATIONLOGGER_H
