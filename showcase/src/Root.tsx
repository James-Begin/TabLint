import {Composition} from 'remotion';
import {TabLintDemo} from './TabLintDemo';
import {timing} from './tabl-int/timing';
export const RemotionRoot: React.FC = () => <Composition id="TabLintDemo" component={TabLintDemo} width={1920} height={1080} fps={30} durationInFrames={timing.totalFrames}/>;
