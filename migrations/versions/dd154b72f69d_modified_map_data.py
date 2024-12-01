"""modified map data

Revision ID: dd154b72f69d
Revises: 129cc7fa6552
Create Date: 2024-11-25 23:43:34.777163

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'dd154b72f69d'
down_revision = '129cc7fa6552'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('tbl_map_data', schema=None) as batch_op:
        batch_op.add_column(sa.Column('cstr_obstacle_shape', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('cflt_midpoint_x_coord', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('cflt_midpoint_y_coord', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('cstr_bottom_left', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('cstr_bottom_right', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('cstr_top_right', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('cstr_top_left', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('cstr_mid_top', sa.String(), nullable=True))
        batch_op.drop_column('cflt_y_coord')
        batch_op.drop_column('cflt_x_coord')
        batch_op.drop_column('cstr_point_type')

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('tbl_map_data', schema=None) as batch_op:
        batch_op.add_column(sa.Column('cstr_point_type', sa.VARCHAR(), autoincrement=False, nullable=True))
        batch_op.add_column(sa.Column('cflt_x_coord', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True))
        batch_op.add_column(sa.Column('cflt_y_coord', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True))
        batch_op.drop_column('cstr_mid_top')
        batch_op.drop_column('cstr_top_left')
        batch_op.drop_column('cstr_top_right')
        batch_op.drop_column('cstr_bottom_right')
        batch_op.drop_column('cstr_bottom_left')
        batch_op.drop_column('cflt_midpoint_y_coord')
        batch_op.drop_column('cflt_midpoint_x_coord')
        batch_op.drop_column('cstr_obstacle_shape')

    # ### end Alembic commands ###
